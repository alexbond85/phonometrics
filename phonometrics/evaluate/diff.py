import re
from difflib import SequenceMatcher

def tokenize_phonemes_with_spaces(phoneme_str):
    # Tokenize the string, including spaces
    tokens = re.findall(r'\S+|\s', phoneme_str)
    return tokens

def remove_spaces(phonemes):
    return [p for p in phonemes if p.strip() != '']

def check_initial_match_percentage(ref_phonemes, user_phonemes):
    ref_phonemes_no_space = remove_spaces(ref_phonemes)
    user_phonemes_no_space = remove_spaces(user_phonemes)

    matcher = SequenceMatcher(None, ref_phonemes_no_space, user_phonemes_no_space)
    matches = sum(triple.size for triple in matcher.get_matching_blocks())
    total = max(len(ref_phonemes_no_space), len(user_phonemes_no_space))
    match_percentage = matches / total

    return match_percentage

def evaluate_and_display(ref_phonemes, ref_probs, user_phonemes, user_probs):
    # Remove spaces for matching
    ref_phonemes_no_space = remove_spaces(ref_phonemes)
    user_phonemes_no_space = remove_spaces(user_phonemes)

    # Align sequences without spaces
    matcher = SequenceMatcher(None, ref_phonemes_no_space, user_phonemes_no_space)
    opcodes = matcher.get_opcodes()

    # Map indices back to original sequences with spaces
    ref_index_map = [i for i, p in enumerate(ref_phonemes) if p.strip() != '']
    user_index_map = [i for i, p in enumerate(user_phonemes) if p.strip() != '']

    differences = []
    total_phonemes = len(ref_phonemes_no_space)
    matching_phonemes = 0
    total_score = 0
    problematic_phonemes = []

    for tag, i1, i2, j1, j2 in opcodes:
        for idx_ref_no_space in range(i1, i2):
            idx_ref = ref_index_map[idx_ref_no_space]
            ref_phoneme = ref_phonemes[idx_ref]
            ref_prob = ref_probs[idx_ref] if idx_ref < len(ref_probs) else None

            if tag == 'equal':
                idx_user_no_space = j1 + (idx_ref_no_space - i1)
                idx_user = user_index_map[idx_user_no_space]
                user_phoneme = user_phonemes[idx_user]
                user_prob = user_probs[idx_user] if idx_user < len(user_probs) else None
                matching_phonemes += 1
                total_score += (ref_prob + user_prob) / 2 if ref_prob and user_prob else 0

                # Determine if the phoneme is problematic
                if user_prob and ref_prob and user_prob < ref_prob * 0.8:
                    problematic_phonemes.append({
                        'phoneme': user_phoneme,
                        'ref_prob': ref_prob,
                        'user_prob': user_prob,
                        'index': idx_ref
                    })

                differences.append({
                    'type': 'match',
                    'ref_phoneme': ref_phoneme,
                    'user_phoneme': user_phoneme,
                    'ref_prob': ref_prob,
                    'user_prob': user_prob,
                    'index': idx_ref
                })
            elif tag == 'replace':
                idx_user_no_space = j1 + (idx_ref_no_space - i1)
                idx_user = user_index_map[idx_user_no_space]
                user_phoneme = user_phonemes[idx_user]
                user_prob = user_probs[idx_user] if idx_user < len(user_probs) else None

                differences.append({
                    'type': 'substitution',
                    'ref_phoneme': ref_phoneme,
                    'user_phoneme': user_phoneme,
                    'ref_prob': ref_prob,
                    'user_prob': user_prob,
                    'index': idx_ref
                })
            elif tag == 'delete':
                differences.append({
                    'type': 'deletion',
                    'ref_phoneme': ref_phoneme,
                    'user_phoneme': '',
                    'ref_prob': ref_prob,
                    'user_prob': None,
                    'index': idx_ref
                })
        if tag == 'insert':
            for idx_user_no_space in range(j1, j2):
                idx_user = user_index_map[idx_user_no_space]
                user_phoneme = user_phonemes[idx_user]
                user_prob = user_probs[idx_user] if idx_user < len(user_probs) else None

                differences.append({
                    'type': 'insertion',
                    'ref_phoneme': '',
                    'user_phoneme': user_phoneme,
                    'ref_prob': None,
                    'user_prob': user_prob,
                    'index': None
                })

    # Calculate final scores
    phoneme_accuracy = matching_phonemes / total_phonemes
    avg_prob = total_score / matching_phonemes if matching_phonemes else 0
    final_score = phoneme_accuracy * avg_prob

    # Build display with spaces
    display_diffs = []
    ref_idx = 0
    diff_idx = 0
    while ref_idx < len(ref_phonemes):
        if ref_phonemes[ref_idx].strip() == '':
            # Space
            display_diffs.append({'type': 'space', 'value': ' '})
            ref_idx += 1
        else:
            if diff_idx < len(differences):
                diff = differences[diff_idx]
                display_diffs.append(diff)
                diff_idx += 1
                ref_idx += 1
            else:
                break  # Should not happen

    return {
        'phoneme_accuracy': phoneme_accuracy,
        'average_probability': avg_prob,
        'final_score': final_score,
        'differences': display_diffs,
        'problematic_phonemes': problematic_phonemes
    }

def display_differences(differences):
    from termcolor import colored

    display_lines = []

    for diff in differences:
        if diff['type'] == 'space':
            display_lines.append(' ')
        elif diff['type'] == 'match':
            display_lines.append(colored(diff['ref_phoneme'], 'green'))
        elif diff['type'] == 'substitution':
            ref_phoneme = colored(diff['ref_phoneme'], 'red', attrs=['strikethrough'])
            user_phoneme = colored(diff['user_phoneme'], 'red')
            display_lines.append(f"{ref_phoneme}->{user_phoneme}")
        elif diff['type'] == 'deletion':
            ref_phoneme = colored(diff['ref_phoneme'], 'yellow', attrs=['strikethrough'])
            display_lines.append(f"{ref_phoneme}(deleted)")
        elif diff['type'] == 'insertion':
            user_phoneme = colored(diff['user_phoneme'], 'blue')
            display_lines.append(f"{user_phoneme}(inserted)")

    display_text = ''.join(display_lines)
    print(display_text)
    return display_text

def plot_phoneme_probabilities(differences):
    import matplotlib.pyplot as plt

    # Prepare data for plotting
    ref_phonemes = []
    ref_probs = []
    user_phonemes = []
    user_probs = []
    indices = []
    x_labels = []

    idx = 0
    for diff in differences:
        if diff['type'] == 'space':
            continue  # Skip spaces in plotting
        else:
            ref_phoneme = diff['ref_phoneme'] if diff['ref_phoneme'] else ''
            user_phoneme = diff['user_phoneme'] if diff['user_phoneme'] else ''
            ref_prob = diff['ref_prob'] if diff['ref_prob'] is not None else 0
            user_prob = diff['user_prob'] if diff['user_prob'] is not None else 0

            ref_phonemes.append(ref_phoneme)
            ref_probs.append(ref_prob)
            user_phonemes.append(user_phoneme)
            user_probs.append(user_prob)
            indices.append(idx)
            x_labels.append(ref_phoneme)  # Use reference phoneme as x-axis label
            idx += 1

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Plot reference phonemes
    axs[0].bar(indices, ref_probs, color='blue')
    axs[0].set_title('Reference Phoneme Probabilities')
    axs[0].set_ylabel('Probability')
    axs[0].set_xticks(indices)
    axs[0].set_xticklabels(x_labels, rotation='vertical')

    # Plot user phonemes
    axs[1].bar(indices, user_probs, color='green')
    axs[1].set_title('User Phoneme Probabilities')
    axs[1].set_ylabel('Probability')
    axs[1].set_xticks(indices)
    axs[1].set_xticklabels(user_phonemes, rotation='vertical')  # Use user phonemes as x-axis labels

    plt.xlabel('Phonemes')
    plt.tight_layout()
    plt.show()

def get_lowest_probability_phonemes(differences, threshold=0.2, top_n=5):
    # Collect phonemes where user_prob is significantly lower than ref_prob
    low_prob_phonemes = []
    for diff in differences:
        if diff['type'] == 'match' and diff['user_prob'] is not None and diff['ref_prob'] is not None:
            prob_diff = diff['ref_prob'] - diff['user_prob']
            if prob_diff > threshold * diff['ref_prob']:
                low_prob_phonemes.append({
                    'phoneme': diff['user_phoneme'],
                    'user_prob': diff['user_prob'],
                    'ref_prob': diff['ref_prob'],
                    'prob_diff': prob_diff
                })

    # Sort by probability difference
    low_prob_phonemes.sort(key=lambda x: x['prob_diff'], reverse=True)

    # Return top N phonemes
    return low_prob_phonemes[:top_n]


def plot_phoneme_probabilities_initial(ref_phonemes, ref_probs, user_phonemes, user_probs):
    import matplotlib.pyplot as plt

    # Remove spaces for plotting
    ref_phonemes_no_space = remove_spaces(ref_phonemes)
    ref_probs_no_space = [prob for phoneme, prob in zip(ref_phonemes, ref_probs) if phoneme.strip() != '']

    user_phonemes_no_space = remove_spaces(user_phonemes)
    user_probs_no_space = [prob for phoneme, prob in zip(user_phonemes, user_probs) if phoneme.strip() != '']

    # Determine the maximum length to handle sequences of different lengths
    max_len = max(len(ref_phonemes_no_space), len(user_phonemes_no_space))

    # Pad the shorter sequence with empty strings or zeros
    ref_phonemes_padded = ref_phonemes_no_space + [''] * (max_len - len(ref_phonemes_no_space))
    ref_probs_padded = ref_probs_no_space + [0] * (max_len - len(ref_probs_no_space))

    user_phonemes_padded = user_phonemes_no_space + [''] * (max_len - len(user_phonemes_no_space))
    user_probs_padded = user_probs_no_space + [0] * (max_len - len(user_probs_no_space))

    indices = range(max_len)

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Plot reference phonemes
    axs[0].bar(indices, ref_probs_padded, color='blue')
    axs[0].set_title('Reference Phoneme Probabilities')
    axs[0].set_ylabel('Probability')
    axs[0].set_xticks(indices)
    axs[0].set_xticklabels(ref_phonemes_padded, rotation='vertical')

    # Plot user phonemes
    axs[1].bar(indices, user_probs_padded, color='green')
    axs[1].set_title('User Phoneme Probabilities')
    axs[1].set_ylabel('Probability')
    axs[1].set_xticks(indices)
    axs[1].set_xticklabels(user_phonemes_padded, rotation='vertical')

    plt.xlabel('Phonemes')
    plt.tight_layout()
    plt.show()

def get_lowest_probability_phonemes_initial(ref_phonemes, ref_probs, user_phonemes, user_probs, threshold=0.2, top_n=5):
    # Remove spaces and align sequences
    ref_phonemes_no_space = remove_spaces(ref_phonemes)
    ref_probs_no_space = [prob for phoneme, prob in zip(ref_phonemes, ref_probs) if phoneme.strip() != '']

    user_phonemes_no_space = remove_spaces(user_phonemes)
    user_probs_no_space = [prob for phoneme, prob in zip(user_phonemes, user_probs) if phoneme.strip() != '']

    # Determine the minimum length to avoid index errors
    min_len = min(len(ref_phonemes_no_space), len(user_phonemes_no_space))

    low_prob_phonemes = []
    for idx in range(min_len):
        user_prob = user_probs_no_space[idx]
        ref_prob = ref_probs_no_space[idx]
        phoneme = user_phonemes_no_space[idx]

        if user_prob < ref_prob * (1 - threshold):
            prob_diff = ref_prob - user_prob
            low_prob_phonemes.append({
                'phoneme': phoneme,
                'user_prob': user_prob,
                'ref_prob': ref_prob,
                'prob_diff': prob_diff
            })

    # Sort by probability difference
    low_prob_phonemes.sort(key=lambda x: x['prob_diff'], reverse=True)

    # Return top N phonemes
    return low_prob_phonemes[:top_n]



def main():
    # Reference data
    reference_phoneme_str = 'ɛl vøt alɛt o kɔ̃sɛʁ mɛz ɛl na pa də bijɛ'
    reference_phonemes = tokenize_phonemes_with_spaces(reference_phoneme_str)
    reference_probs = [0.8925, 0.9995, 0.7856, 0.9995, 0.9045, 0.6495, 0.6000, 0.9943,
                       0.9982, 0.4823, 0.7263, 0.9966, 0.9961, 0.9933, 0.9998, 0.9191,
                       0.9126, 0.9988, 0.9134, 0.9038, 0.9991, 0.9983, 0.8942, 0.9413,
                       0.9953, 0.7878, 0.9969, 0.9959, 0.9995, 0.9992, 0.9952, 0.9986,
                       0.9990, 0.7643, 0.9991, 0.9979, 0.9971, 0.8949, 0.9931, 0.9812,
                       0.8580]

    # User data
    # user_phoneme_str = 'øl vø alɛt o kosɛʁ mɛ ɛl na pa də bia'
    user_phoneme_str = 'øl vø alɛt o kosɛʁ mɛ ɛl na pa də bia'
    #
    user_phonemes = tokenize_phonemes_with_spaces(user_phoneme_str)
    user_probs = [0.8925, 0.6000, 0.7856, 0.5000, 0.9045, 0.6000, 0.6000, 0.5000,
                  0.5000, 0.4823, 0.7263, 0.9900, 0.9900, 0.9900, 0.5000, 0.9191,
                  0.9126, 0.9900, 0.9134, 0.9038, 0.5000, 0.9900, 0.8942, 0.9413,
                  0.9900, 0.7878, 0.9900, 0.9900, 0.9900, 0.9900, 0.9900, 0.9900,
                  0.9900, 0.7643, 0.9900, 0.9900, 0.9900, 0.8949, 0.9900, 0.9800]

    # Check initial match percentage
    match_percentage = check_initial_match_percentage(reference_phonemes, user_phonemes)
    print(f"Initial phoneme match percentage: {match_percentage * 100:.2f}%")

    # Evaluate and display
    result = evaluate_and_display(reference_phonemes, reference_probs, user_phonemes, user_probs)
    display_differences(result['differences'])

    # Plot phoneme probabilities
    plot_phoneme_probabilities(result['differences'])

    plot_phoneme_probabilities_initial(reference_phonemes, reference_probs, user_phonemes, user_probs)

    # Get phonemes with lowest probabilities
    low_prob_phonemes = get_lowest_probability_phonemes(result['differences'], threshold=0.2, top_n=5)
    print("\nPhonemes with lowest probabilities (user vs. reference):")
    for p in low_prob_phonemes:
        print(f"Phoneme: {p['phoneme']}, User Prob: {p['user_prob']:.4f}, Ref Prob: {p['ref_prob']:.4f}, Diff: {p['prob_diff']:.4f}")

if __name__ == "__main__":
    main()
