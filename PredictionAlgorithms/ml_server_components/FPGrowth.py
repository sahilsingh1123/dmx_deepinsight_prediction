import pyfpgrowth
import time


def perform_fp_growth(data):
    start_time = time.time()
    print(len(data))
    patterns = pyfpgrowth.find_frequent_patterns(data, int(len(data) * 0.1))
    # print(sorted(patterns.items(), key=lambda x: x[1], reverse=True))
    frequent_pattern_list = []
    frequent_frequency_list = []
    count = 0
    for pattern in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
        # print(pattern[0])
        # print(pattern[1])
        if count > 99:
            break
        if pattern[1] > 1:
            frequent_pattern_list.append(pattern[0])
            frequent_frequency_list.append(pattern[1])
            count += 1
    json_response = {'run_status': 'success', 'frequentPatternList': frequent_pattern_list,
                     'frequentFrequencyList': frequent_frequency_list, 'execution_time': time.time() - start_time}
    # print(json.dumps(json_response))
    print('Found Patterns!')
    #return str(json.dumps(json_response)).encode('utf-8')
    return json_response

# patterns = pyfpgrowth.find_frequent_patterns(transactions, 4)
# print(patterns)
# rules = pyfpgrowth.generate_association_rules(patterns, 0.9)
# print(rules)
# perform_fp_growth(transactions)
