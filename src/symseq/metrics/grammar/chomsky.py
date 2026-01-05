# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, symseq Contributors

"""Chomsky hierarchy classification via pattern detection."""

import numpy as np
import math
from collections import Counter, defaultdict
from gzip import compress
from tqdm import tqdm


def chomsky_classification(sequence: list[str], verbose: bool = False, 
                          max_search_positions: int = 500, max_pairs: int = 1000) -> dict:
    """
    Classify sequence position in Chomsky hierarchy.

    Parameters
    ----------
    sequence : list of str
        Symbolic sequence.
    verbose : bool, default=False
        If True, display progress information during classification.
    max_search_positions : int, default=500
        Maximum number of positions to search for triple dependencies (A^nB^nC^n).
        Limits computational complexity on long sequences.
    max_pairs : int, default=1000
        Maximum number of pairs to consider for crossing dependencies.
        Limits computational complexity on long sequences.

    Returns
    -------
    dict
        Classification results with keys:
        - 'type': Estimated Chomsky type (0-3)
        - 'classification': Human-readable classification name
        - 'confidence': Confidence in classification
        - 'scores': Scores for each type
        - 'patterns': Detected pattern details
        - 'evidence': Interpretive evidence for classification
        - 'reliable': Whether classification is reliable

    Notes
    -----
    Type 3 (Regular): FSA-recognizable, periodic patterns
    Type 2 (Context-Free): Nested structures, A^nB^n
    Type 1 (Context-Sensitive): A^nB^nC^n, crossing dependencies
    Type 0 (Unrestricted): High Kolmogorov complexity

    References
    ----------
    Chomsky, N. (1956). Three models for the description of language.
    IRE Transactions on Information Theory, 2(3), 113-124.
    """
    if len(sequence) < 3:
        return {
            'type': None,
            'confidence': 0.0,
            'scores': {},
            'patterns': {},
            'reliable': False
        }

    if verbose:
        print(f"Classifying sequence of length {len(sequence)} in Chomsky hierarchy...")
        print("Detecting patterns for each hierarchy level...")

    # Type 3 (Regular) patterns
    if verbose:
        print("  Checking Type 3 (Regular) patterns...")
    type3_patterns = {}
    type3_detectors = [
        ('repetition', _detect_repetition),
        ('alternation', _detect_alternation),
        ('periodicity', _detect_cycle),
        ('fsa_determinism', _detect_finite_state),
    ]
    for name, detector in (tqdm(type3_detectors, desc="Type 3 patterns", disable=not verbose) if verbose else type3_detectors):
        type3_patterns[name] = detector(sequence)

    # Type 2 (Context-Free) patterns
    if verbose:
        print("  Checking Type 2 (Context-Free) patterns...")
    type2_patterns = {}
    type2_detectors = [
        ('balanced_pairs', _detect_balanced_parentheses),
        ('center_embedding', _detect_center_embedding),
        ('nested_structures', _detect_nested_structures),
        ('palindromes', _detect_palindromes),
    ]
    for name, detector in (tqdm(type2_detectors, desc="Type 2 patterns", disable=not verbose) if verbose else type2_detectors):
        type2_patterns[name] = detector(sequence)

    # Type 1 (Context-Sensitive) patterns
    if verbose:
        print("  Checking Type 1 (Context-Sensitive) patterns...")
    type1_patterns = {}
    type1_detectors = [
        ('triple_dependencies', lambda seq: _detect_triple_dependencies(seq, max_search_positions)),
        ('crossing_dependencies', lambda seq: _detect_crossing_dependencies(seq, max_pairs)),
        ('length_sensitive', _detect_length_sensitive),
        ('context_dependent', _detect_context_dependent),
    ]
    for name, detector in (tqdm(type1_detectors, desc="Type 1 patterns", disable=not verbose) if verbose else type1_detectors):
        type1_patterns[name] = detector(sequence)

    # Type 0 (Unrestricted) patterns
    if verbose:
        print("  Checking Type 0 (Unrestricted) patterns...")
    # Note: Kolmogorov complexity returns compressed_size/original_size
    # LOW ratio (e.g., 0.15) = highly compressible = Type 3 (regular)
    # HIGH ratio (e.g., 0.9) = incompressible = Type 0 (random/complex)
    kolmogorov = _estimate_kolmogorov_complexity(sequence)
    apen = _calculate_approximate_entropy(sequence)
    
    # Type 0 should have HIGH compression ratio (incompressible)
    # But also consider that true randomness has high ApEn
    # Regular processes have LOW compression ratio AND moderate ApEn
    type0_patterns = {
        'incompressibility': kolmogorov,  # High ratio = incompressible
        'irregularity': apen,  # High ApEn = unpredictable
    }

    type3_score = np.mean(list(type3_patterns.values()))
    type2_score = np.mean(list(type2_patterns.values()))
    type1_score = np.mean(list(type1_patterns.values()))
    type0_score = np.mean(list(type0_patterns.values()))
    
    # Boost Type 3 if FSA determinism is high (Markov chains should score high here)
    if type3_patterns.get('fsa_determinism', 0) > 0.6:
        type3_score = min(1.0, type3_score * 1.3)

    scores = {
        'type3': type3_score,
        'type2': type2_score,
        'type1': type1_score,
        'type0': type0_score,
    }
    
    if verbose:
        print(f"\n  Raw scores: Type3={type3_score:.3f}, Type2={type2_score:.3f}, Type1={type1_score:.3f}, Type0={type0_score:.3f}")
        print(f"  Key patterns: FSA_det={type3_patterns.get('fsa_determinism', 0):.3f}, Kolmogorov={kolmogorov:.3f}, ApEn={apen:.3f}")

    adjusted_scores = scores.copy()
    
    # Hierarchy enforcement: prefer simpler classes
    if type3_score > 0.5:  # Lowered threshold for Type 3
        adjusted_scores['type2'] *= 0.5
        adjusted_scores['type1'] *= 0.5
        adjusted_scores['type0'] *= 0.3  # Strong penalty
    if type2_score > 0.4:
        adjusted_scores['type0'] *= 0.5
        adjusted_scores['type1'] *= 0.7
    if type1_score > 0.4:
        adjusted_scores['type0'] *= 0.5

    max_score = max(adjusted_scores.values())
    estimated_type = max(adjusted_scores.keys(), key=lambda k: adjusted_scores[k])
    estimated_type_num = int(estimated_type.replace('type', ''))

    reliable = max_score > 0.3 and len(sequence) > 20
    
    # Build evidence string
    type_names = {3: 'Regular (Type 3)', 2: 'Context-Free (Type 2)', 
                  1: 'Context-Sensitive (Type 1)', 0: 'Unrestricted (Type 0)'}
    evidence_parts = []
    if estimated_type_num == 3:
        if type3_patterns.get('periodicity', 0) > 0.5:
            evidence_parts.append('Strong periodic patterns')
        if type3_patterns.get('fsa_determinism', 0) > 0.5:
            evidence_parts.append('Deterministic FSA transitions')
    elif estimated_type_num == 2:
        if type2_patterns.get('balanced_pairs', 0) > 0.3:
            evidence_parts.append('Balanced A^nB^n patterns')
        if type2_patterns.get('nested_structures', 0) > 0.3:
            evidence_parts.append('Nested structures')
    elif estimated_type_num == 1:
        if type1_patterns.get('triple_dependencies', 0) > 0.3:
            evidence_parts.append('A^nB^nC^n patterns')
        if type1_patterns.get('crossing_dependencies', 0) > 0.3:
            evidence_parts.append('Crossing dependencies')
    else:
        if type0_patterns.get('incompressibility', 0) > 0.7:
            evidence_parts.append('High incompressibility (poor compression)')
        if type0_patterns.get('irregularity', 0) > 0.5:
            evidence_parts.append('High irregularity (unpredictable)')
    
    evidence = '; '.join(evidence_parts) if evidence_parts else 'No strong patterns detected'

    return {
        'type': estimated_type_num,
        'classification': type_names.get(estimated_type_num, f'Type {estimated_type_num}'),
        'confidence': max_score,
        'scores': scores,
        'patterns': {
            'type3': type3_patterns,
            'type2': type2_patterns,
            'type1': type1_patterns,
            'type0': type0_patterns,
        },
        'evidence': evidence,
        'reliable': reliable
    }


def _detect_repetition(sequence: list[str]) -> float:
    """Detect simple repetition patterns."""
    if len(sequence) < 2:
        return 0.0

    best_confidence = 0.0

    for unit_length in range(1, min(len(sequence) // 2 + 1, 10)):
        for start in range(min(unit_length, len(sequence))):
            unit = ''.join(sequence[start:start + unit_length])
            if not unit:
                continue

            matches = 0
            total_possible = 0
            i = start
            while i + unit_length <= len(sequence):
                total_possible += 1
                current_unit = ''.join(sequence[i:i + unit_length])
                if current_unit == unit:
                    matches += 1
                i += unit_length

            if total_possible > 0:
                confidence = matches / total_possible
                if confidence > best_confidence and matches >= 2:
                    best_confidence = confidence

    return best_confidence


def _detect_alternation(sequence: list[str]) -> float:
    """Detect alternation patterns (ABAB, etc.)."""
    if len(sequence) < 4:
        return 0.0

    best_confidence = 0.0

    for pattern_length in range(2, min(len(sequence) // 2 + 1, 8)):
        pattern = sequence[:pattern_length]
        matches = 0
        total_checks = 0

        for i in range(0, len(sequence) - pattern_length + 1, pattern_length):
            total_checks += 1
            current_segment = sequence[i:i + pattern_length]
            if current_segment == pattern:
                matches += 1

        if total_checks > 0:
            confidence = matches / total_checks
            if confidence > best_confidence and matches >= 2:
                best_confidence = confidence

    return best_confidence


def _detect_cycle(sequence: list[str]) -> float:
    """Detect cyclic patterns with period detection."""
    if len(sequence) < 3:
        return 0.0

    best_confidence = 0.0

    for period in range(1, min(len(sequence) // 2 + 1, 10)):
        matches = 0
        total_checks = 0

        for i in range(period, len(sequence)):
            total_checks += 1
            if sequence[i] == sequence[i % period]:
                matches += 1

        if total_checks > 0:
            confidence = matches / total_checks
            if confidence > best_confidence and matches >= period:
                best_confidence = confidence

    return best_confidence


def _detect_finite_state(sequence: list[str]) -> float:
    """Detect FSA patterns via transition determinism."""
    if len(sequence) < 2:
        return 0.0

    transitions = defaultdict(Counter)
    for i in range(len(sequence) - 1):
        transitions[sequence[i]][sequence[i + 1]] += 1

    total_entropy = 0.0
    max_possible_entropy = 0.0
    num_states = len(transitions)

    for state, next_states in transitions.items():
        total_transitions = sum(next_states.values())
        if total_transitions > 0:
            probs = [count / total_transitions for count in next_states.values()]
            state_entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            total_entropy += state_entropy

            max_entropy = math.log2(len(next_states)) if len(next_states) > 1 else 0
            max_possible_entropy += max_entropy

    avg_entropy = total_entropy / num_states if num_states > 0 else 0
    max_avg_entropy = max_possible_entropy / num_states if num_states > 0 else 1
    determinism = 1 - (avg_entropy / max_avg_entropy) if max_avg_entropy > 0 else 1

    return determinism


def _detect_balanced_parentheses(sequence: list[str]) -> float:
    """Detect A^n B^n patterns."""
    if len(sequence) < 4:
        return 0.0

    best_confidence = 0.0
    unique_symbols = list(set(sequence))

    if len(unique_symbols) < 2:
        return 0.0

    matches = []
    
    for symbol_a in unique_symbols:
        for symbol_b in unique_symbols:
            if symbol_a == symbol_b:
                continue

            for start in range(len(sequence) - 3):
                n_a = 0
                i = start
                while i < len(sequence) and sequence[i] == symbol_a:
                    n_a += 1
                    i += 1

                if n_a == 0:
                    continue

                n_b = 0
                while i < len(sequence) and sequence[i] == symbol_b:
                    n_b += 1
                    i += 1

                if n_a == n_b and n_a >= 2:
                    matches.append(n_a)

    if matches:
        avg_match_length = np.mean(matches)
        num_matches = len(matches)
        coverage = sum(matches) * 2 / len(sequence)
        best_confidence = min(1.0, (coverage + num_matches / 10) / 2)

    return best_confidence


def _detect_center_embedding(sequence: list[str]) -> float:
    """Detect center-embedded structures."""
    if len(sequence) < 3:
        return 0.0

    best_confidence = 0.0

    for length in range(3, min(len(sequence) + 1, 20)):
        for start in range(len(sequence) - length + 1):
            subseq = sequence[start:start + length]
            if subseq == subseq[::-1]:
                confidence = length / len(sequence)
                best_confidence = max(best_confidence, confidence)

    return best_confidence


def _detect_nested_structures(sequence: list[str]) -> float:
    """Detect nested structures using stack-based parsing."""
    if len(sequence) < 4:
        return 0.0

    unique_symbols = list(set(sequence))
    if len(unique_symbols) < 2:
        return 0.0

    best_confidence = 0.0

    for open_sym in unique_symbols:
        for close_sym in unique_symbols:
            if open_sym == close_sym:
                continue

            stack = []
            matched = 0
            total = 0

            for symbol in sequence:
                if symbol == open_sym:
                    stack.append(symbol)
                    total += 1
                elif symbol == close_sym:
                    total += 1
                    if stack:
                        stack.pop()
                        matched += 1

            if total > 0:
                confidence = matched / total
                best_confidence = max(best_confidence, confidence)

    return best_confidence


def _detect_palindromes(sequence: list[str]) -> float:
    """Detect palindromic patterns."""
    return _detect_center_embedding(sequence)


def _detect_triple_dependencies(sequence: list[str], max_search_positions: int = 500) -> float:
    """Detect A^n B^n C^n patterns.
    
    Parameters
    ----------
    sequence : list of str
        Symbolic sequence.
    max_search_positions : int, default=500
        Maximum number of positions to search. Limits computational complexity.
    
    Returns
    -------
    float
        Confidence score [0, 1] for triple dependency patterns.
    """
    if len(sequence) < 6:
        return 0.0

    unique_symbols = list(set(sequence))
    if len(unique_symbols) < 3:
        return 0.0

    # Limit search to avoid O(n^4) complexity on long sequences
    max_search_positions = min(len(sequence), max_search_positions)
    search_step = max(1, len(sequence) // max_search_positions)
    
    matches = []

    for sym_a in unique_symbols:
        for sym_b in unique_symbols:
            for sym_c in unique_symbols:
                if len(set([sym_a, sym_b, sym_c])) < 3:
                    continue

                for start in range(0, len(sequence) - 5, search_step):
                    i = start
                    n_a = 0
                    while i < len(sequence) and sequence[i] == sym_a:
                        n_a += 1
                        i += 1

                    if n_a == 0 or n_a > 20:  # Skip very long runs
                        continue

                    n_b = 0
                    while i < len(sequence) and sequence[i] == sym_b:
                        n_b += 1
                        i += 1

                    n_c = 0
                    while i < len(sequence) and sequence[i] == sym_c:
                        n_c += 1
                        i += 1

                    if n_a == n_b == n_c and n_a >= 2:
                        matches.append(n_a)
                        if len(matches) >= 10:  # Early termination
                            break
                if len(matches) >= 10:
                    break
            if len(matches) >= 10:
                break
        if len(matches) >= 10:
            break

    if matches:
        num_matches = len(matches)
        coverage = sum(matches) * 3 / len(sequence)
        best_confidence = min(1.0, (coverage + num_matches / 5) / 2)
        return best_confidence

    return 0.0


def _detect_crossing_dependencies(sequence: list[str], max_pairs: int = 1000) -> float:
    """Detect crossing (non-nested) dependencies.
    
    Parameters
    ----------
    sequence : list of str
        Symbolic sequence.
    max_pairs : int, default=1000
        Maximum number of pairs to consider. Limits computational complexity.
    
    Returns
    -------
    float
        Confidence score [0, 1] for crossing dependency patterns.
    """
    if len(sequence) < 4:
        return 0.0

    # Limit to avoid O(n^2) on very long sequences
    pairs = []
    for i in range(min(len(sequence), 1000)):
        for j in range(i + 1, min(len(sequence), i + 100)):
            if sequence[i] == sequence[j]:
                pairs.append((i, j))
                if len(pairs) >= max_pairs:
                    break
        if len(pairs) >= max_pairs:
            break

    if len(pairs) < 2:
        return 0.0

    crossing_pairs = 0
    total_pairs = 0
    max_comparisons = 5000

    for i, (a1, a2) in enumerate(pairs):
        for b1, b2 in pairs[i + 1:]:
            total_pairs += 1
            if a1 < b1 < a2 < b2:
                crossing_pairs += 1
            if total_pairs >= max_comparisons:
                break
        if total_pairs >= max_comparisons:
            break

    return crossing_pairs / total_pairs if total_pairs > 0 else 0.0


def _detect_length_sensitive(sequence: list[str]) -> float:
    """Detect length-sensitive patterns."""
    if len(sequence) < 5:
        return 0.0

    positions = defaultdict(list)
    for i, symbol in enumerate(sequence):
        positions[symbol].append(i)

    correlations = []
    for symbol, pos_list in positions.items():
        if len(pos_list) > 1:
            corr = np.corrcoef(pos_list, range(len(pos_list)))[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))

    return np.mean(correlations) if correlations else 0.0


def _detect_context_dependent(sequence: list[str]) -> float:
    """Detect context-dependent production rules."""
    if len(sequence) < 4:
        return 0.0

    context_transitions = defaultdict(lambda: defaultdict(int))

    for i in range(1, len(sequence) - 1):
        left_context = sequence[i - 1]
        current = sequence[i]
        right_context = sequence[i + 1]
        context_transitions[(left_context, current)][right_context] += 1

    context_dependencies = []
    for (left, curr), next_counts in context_transitions.items():
        if sum(next_counts.values()) > 1:
            probs = [count / sum(next_counts.values()) for count in next_counts.values()]
            entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            max_entropy = math.log2(len(next_counts))
            determinism = 1 - (entropy / max_entropy) if max_entropy > 0 else 0
            context_dependencies.append(determinism)

    return np.mean(context_dependencies) if context_dependencies else 0.0


def _estimate_kolmogorov_complexity(sequence: list[str]) -> float:
    """Estimate Kolmogorov complexity via compression."""
    if not sequence:
        return 0.0

    string_repr = ''.join(sequence)
    original_size = len(string_repr)
    compressed_size = len(compress(string_repr.encode('utf-8')))

    complexity = compressed_size / original_size if original_size > 0 else 0
    return min(complexity, 1.0)


def _calculate_approximate_entropy(sequence: list[str], m: int = 2, r: float = 0.2) -> float:
    """Calculate approximate entropy (ApEn) as chaos indicator."""
    if len(sequence) < m + 1:
        return 0.0

    def _phi(m_val):
        patterns = []
        for i in range(len(sequence) - m_val + 1):
            patterns.append(tuple(sequence[i:i + m_val]))

        pattern_counts = Counter(patterns)
        n = len(patterns)

        phi_val = 0.0
        for count in pattern_counts.values():
            if count > 0:
                phi_val += (count / n) * math.log(count / n)

        return phi_val

    phi_m = _phi(m)
    phi_m_plus_1 = _phi(m + 1)

    apen = phi_m - phi_m_plus_1
    return min(abs(apen), 1.0)
