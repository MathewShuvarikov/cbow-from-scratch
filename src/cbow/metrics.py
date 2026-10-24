import numpy as np

def theoretical_loss_bound(
    training_pairs: list[tuple[list[int], int]]
) -> float:

    context_targets = {}
    for context, target_id in training_pairs:
        context_key = tuple(sorted(context))
        if context_key not in context_targets:
            context_targets[context_key] = [target_id]
        else:
            context_targets[context_key].append(target_id)

    frequencies = {}
    for context_key, target_ids in context_targets.items():
        sub_dict = {}
        for i in target_ids:
            if i not in sub_dict:
                sub_dict[i] = 1
            else:
                sub_dict[i] += 1

        frequencies[context_key] = sub_dict

    loss_bound = []

    for context_key, frequency in frequencies.items():
        n_c = sum(frequency.values())
        context_loss = 0
        for n_c_t in frequency.values():
            context_loss += -n_c_t*np.log(n_c_t/n_c)
        loss_bound.append(context_loss)



    return float(sum(loss_bound)/len(training_pairs))