from BayesNet import BayesNet
from BNReasoner import BNReasoner
from os.path import exists
from NetworkGenerator import NetworkGenerator
from tqdm import tqdm

import csv, datetime, logging, random, time

if __name__ == '__main__':
    ng = NetworkGenerator()

    run_start = datetime.datetime.now()

    # create csv file if it does not exist
    if not exists('results.csv'):
        header = ['timestamp', 'nodes', 'edges', 'evidence', 'query', 'algorithm', 'order', 'time']
        with open('results.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    for i in tqdm(range(70, 71, 10)):
        for _ in range(3):
            bn = ng.generate_network(i)
            br = BNReasoner(bn, log_level=logging.CRITICAL)

            num_of_edges = sum([len(bn.get_children(var)) for var in [var for var in bn.get_all_variables()]])

            # create random evidence
            evidence = dict()
            evidence_vars = random.choices(list(bn.get_all_variables()), k=int(i/10))
            
            for var in evidence_vars:
                evidence[var] = random.choice([True, False])

            # run MPE
            for order in [br.order_random, br.order_min_degree, br.order_min_fill]:
                start_time = time.time()
                mpe_random = br.MPE(evidence, order_function=order)
                run_time = time.time() - start_time

                with open('results.csv', 'a+', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([run_start, i, num_of_edges, len(evidence), 0, 'MPE', order.__name__, run_time])

            # test MAP
            query_vars = random.choices(list(bn.get_all_variables()), k=int(i/10))
            for order in [br.order_random, br.order_min_degree, br.order_min_fill]:
                start_time = time.time()
                mpe_random = br.MAP(query_vars, evidence, order_function=order)
                run_time = time.time() - start_time

                with open('results.csv', 'a+', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([run_start, i, num_of_edges, len(evidence), len(query_vars), 'MAP', order.__name__, run_time])