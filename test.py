from new_BayesNet import BayesNet
from new_BNReasoner import BNReasoner

if __name__ == '__main__':
    bn = BayesNet()
    bn.load_from_bifxml('testing/lecture_example.BIFXML')
    bnr = BNReasoner(bn)

    #bn.draw_structure()

    # Test pruning
    if False:
        new_bn = bnr.prune(['Wet Grass?'], {'Rain?': False})
        new_bn.draw_structure()

    # Test d-separation
    if False:
        pass

    # Test order_min_degree
    if False:
        print(bnr.order_min_degree())

    # Test order_min_fill
    if False:
        print(bnr.order_min_fill())

    # Test order_random
    if False:
        print(bnr.order_random())

    # Test conditioning
    if False:
        print(bnr.condition(bn.get_cpt('Wet Grass?'), {'Rain?': False}))

    # Test summing_out
    if False:
        print(bnr.summing_out(bn.get_cpt('Winter?'), ['Winter?']))

    # Test maxing_out
    if False:
        assignment = dict()
        print(bnr.maxing_out(bn.get_cpt('Winter?'), ['Winter?'], assignment))
        print(assignment)

    # Test MPE
    if False:
        print('Random:')
        for i in range(10): print(bnr.MPE({'Wet Grass?': False}, bnr.order_random))
        print('Min Degree:')
        for i in range(10): print(bnr.MPE({'Wet Grass?': False}, bnr.order_min_degree))
        print('Min Fill:')
        for i in range(10): print(bnr.MPE({'Wet Grass?': False}, bnr.order_min_fill))

    # Test MAP
    if True:
        print('Random:')
        for i in range(10): print(bnr.MAP(['Sprinkler?', 'Winter?'], {'Wet Grass?': False}, bnr.order_random))
        print('Min Degree:')
        for i in range(10): print(bnr.MAP(['Sprinkler?', 'Winter?'], {'Wet Grass?': False}, bnr.order_min_degree))
        print('Min Fill:')
        for i in range(10): print(bnr.MAP(['Sprinkler?', 'Winter?'], {'Wet Grass?': False}, bnr.order_min_fill))