import numpy as np
import sys

def evaluate(model, attack, provider, start=None, end=None, deterministic=False, debug=False):
    '''
    Evaluate an attack on a particular model and return attack success rate.

    An attack is allowed to be adaptive, so it's fine to design the attack
    based on the specific model it's supposed to break.

    `start` (inclusive) and `end` (exclusive) are indices to evaluate on. If
    unspecified, evaluates on the entire dataset.

    `deterministic` specifies whether to seed the RNG with a constant value for
    a more deterministic test (so randomly selected target classes are chosen
    in a pseudorandom way).
    '''

    if not provider.provides(model.dataset):
        raise ValueError('provider does not provide correct dataset')
    if start is not None and not (0 <= start < len(provider)):
        raise ValueError('start value out of range')
    if end is not None and not (0 <= end <= len(provider)):
        raise ValueError('end value out of range')

    threat_model = model.threat_model
    targeted = threat_model.targeted

    success = 0
    total = 0
    for i in range(start, end):
        print('evaluating %d of [%d, %d)' % (i, start, end), file=sys.stderr)
        total += 1
        x, y = provider[i]
        target = None
        if targeted:
            target = choose_target(i, y, model.dataset.labels, deterministic)
        x_adv = attack.run(np.copy(x), y, target)
        if not threat_model.check(np.copy(x), np.copy(x_adv)):
            if debug:
                print('check failed', file=sys.stderr)
            continue
        y_adv = model.classify(np.copy(x_adv))
        if debug:
            print('true = %d, adv = %d' % (y, y_adv), file=sys.stderr)
        if targeted:
            if y_adv == target:
                success += 1
        else:
            if y_adv != y:
                success += 1

    success_rate = success / total

    return success_rate

def choose_target(index, true_label, num_labels, deterministic=False):
    if deterministic:
        rng = np.random.RandomState(index)
    else:
        rng = np.random.RandomState()

    target = true_label
    while target == true_label:
        target = rng.randint(0, num_labels)

    return target
