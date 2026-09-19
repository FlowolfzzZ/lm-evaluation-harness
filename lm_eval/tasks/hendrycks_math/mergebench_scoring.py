"""Final-answer extraction and normalized exact match for local math tasks."""

import re
from fractions import Fraction

from lm_eval.tasks.hendrycks_math.utils import process_docs as math_process_docs
from lm_eval.tasks.hendrycks_math.utils import strip_string


_ANSWER_MARKER = re.compile(
    r'\b(?:the\s+)?(?:final\s+)?(?:answer|result)\s*(?:is\s*:?\s*|:\s*)'
    r'|####\s*|^\s*(?:actually|correction)\s*:?\s+',
    re.IGNORECASE | re.MULTILINE,
)
_MATH_SPAN = re.compile(r'\$\$(.*?)\$\$|\$([^$]+)\$|\\\((.*?)\\\)|\\\[(.*?)\\\]', re.DOTALL)
_NUMBER = re.compile(r'-?\$?(?:\d[\d,]*(?:\.\d*)?|\.\d+)')
_UNIT = (
    r'(?:dollars?|cents?|USD|eggs?|hours?|minutes?|seconds?|days?|weeks?|months?|years?'
    r'|meters?|miles?|kilometers?|feet|pounds?|kg|km|cm|m|%)'
    r'(?:\s+per\s+(?:day|hour|minute|second))?'
)


def _clean(answer):
    answer = answer.strip().rstrip('.').strip()
    for left, right in [('$$', '$$'), ('$', '$'), (r'\(', r'\)'), (r'\[', r'\]')]:
        if answer.startswith(left) and answer.endswith(right):
            answer = answer[len(left):-len(right)].strip()
            break
    return answer


def _last_box(text):
    """Return (present, value); an incomplete final box must not use an earlier box."""
    boxes = list(re.finditer(r'\\(?:boxed|fbox)\s*', text))
    if not boxes:
        return False, None
    start = boxes[-1].end()
    if start == len(text):
        return True, None
    if text[start] != '{':
        return True, None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == '{':
            depth += 1
        elif text[index] == '}':
            depth -= 1
            if depth == 0:
                raw_tail = text[index + 1:]
                tail = re.sub(r'^(?:\s|[.$]|\\\)|\\\])+', '', raw_tail)
                if tail and tail != 'I hope it is correct.':
                    # A later line overrides an intermediate box. A same-line
                    # expression after the box means the box is only a subexpression.
                    return (False, None) if '\n' in raw_tail else (True, None)
                return True, text[start + 1:index].strip() or None
    return True, None


def extract_final_answer(response):
    """Prefer an explicit final-answer marker, then the last box or final line."""
    markers = list(_ANSWER_MARKER.finditer(response))
    text = response[markers[-1].end():] if markers else response
    lines = [line.strip() for line in text.splitlines() if line.strip() and line.strip() != 'I hope it is correct.']
    if not lines:
        return None
    line = lines[-1]
    if re.match(r'(?:not|incorrect|wrong)\b', text.lstrip() if markers else line, re.IGNORECASE):
        return None
    present, boxed = _last_box(text)
    if present:
        return boxed
    spans = list(_MATH_SPAN.finditer(line))
    if len(spans) == 1:
        prefix = line[:spans[0].start()].strip()
        suffix = line[spans[0].end():].strip().rstrip('.').strip()
        prefix_ok = prefix in ('', '-') or re.fullmatch(
            r'(?:we find|thus|therefore|hence|so)\s*[:,]?', prefix, re.IGNORECASE
        )
        suffix_ok = not suffix or re.fullmatch(_UNIT, suffix, re.IGNORECASE)
        if prefix_ok and suffix_ok:
            line = next(group for group in spans[0].groups() if group is not None)
            if prefix == '-':
                line = '-' + line
    elif spans:
        # Preserve multi-value answers such as "$4$, $6$, $14$, $15$".
        line = line.replace('$', '')
    return _clean(line) or None


def _numeric_value(answer, allow_thousands=True):
    answer = _clean(answer).replace('$', '')
    if ',' in answer and not allow_thousands:
        return None
    answer = answer.replace(',', '')
    fraction = re.fullmatch(r'\\(?:frac|dfrac|tfrac)\{(-?[\d.]+)\}\{(-?[\d.]+)\}', answer)
    try:
        if fraction:
            return Fraction(fraction[1]) / Fraction(fraction[2])
        return Fraction(answer)
    except (ValueError, ZeroDivisionError):
        return None


def gsm8k_process_results(doc, results):
    answer = extract_final_answer(results[0])
    target = doc['answer'].split('####')[-1].strip()
    prediction = _numeric_value(answer) if answer else None
    if prediction is None and answer:
        # A single numeric answer may carry a unit, e.g. "18 dollars".
        unit_answer = re.fullmatch(
            '(' + _NUMBER.pattern + r')\s+' + _UNIT, answer.replace('$', ''), re.IGNORECASE
        )
        if unit_answer:
            prediction = _numeric_value(unit_answer[1])
    gold = _numeric_value(target)
    return {'exact_match': int(prediction is not None and gold is not None and prediction == gold)}


def _normalize_math(answer):
    # Keep the original MATH string normalization, including LaTeX fractions.
    answer = re.sub(r'^[A-Za-z]\s*\\in\s*', '', _clean(answer))
    return strip_string(answer)


def math_process_results(doc, results):
    answer = extract_final_answer(results[0])
    if not answer:
        return {'exact_match': 0}
    target = doc['answer']
    # In MATH commas can delimit lists/tuples, so do not erase them as thousands.
    prediction = _numeric_value(answer, allow_thousands=False)
    gold = _numeric_value(target, allow_thousands=False)
    if prediction is not None and gold is not None:
        match = prediction == gold
    else:
        try:
            match = _normalize_math(answer) == _normalize_math(target)
        except (AssertionError, IndexError, ValueError):
            match = False
    return {'exact_match': int(match)}
