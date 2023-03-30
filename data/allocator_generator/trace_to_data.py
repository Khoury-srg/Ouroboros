import json
import argparse
import os


def parse():
    parser = argparse.ArgumentParser(
        description=f'Translate unordered stack trace log to lifetime training data. Example usage: {__file__} example_data/trace_log.txt example_data/train_data')
    parser.add_argument('trace_log', type=str,
                        help='The file name of the trace log')
    parser.add_argument('output_pref', type=str,
                        help='The file name prefix of the output training datas')
    parser.add_argument('dim', type=int,
                        help='The number of dimensions (pad 0 if the stack is shorter, truncate if the stack is longer)')

    args = parser.parse_args()
    return args


def main(args):
    allocated = {}
    callstack_lookup = {}
    lookup_i = 1

    output = open(args.output_pref+"_raw.txt", 'a')
    encoded_output = open(args.output_pref+'_encoded.txt', 'a')
    if os.path.exists(args.output_pref+'_table.json'):
        lookup_table_f = open(args.output_pref+'_table.json', 'r')
        tmp = json.load(lookup_table_f)
        callstack_lookup = tmp['table']
        lookup_i = tmp['i']
        lookup_table_f.close()
    dim = args.dim

    max_stack_len = -1
    with open(args.trace_log, 'r') as f:
        while True:
            l = f.readline()
            if not l:
                break
            l = l.replace('\n', '')
            l = l.split()
            if l[0] == 'started':
                continue
            if l[0] == 'malloc':
                trace = []
                seen_main = False
                while True:
                    line = f.readline().replace('\n', '').split()
                    if line[0] == "end":
                        break
                    if seen_main:
                        continue
                    # stacklayer = ''.join(line[3:])
                    stacklayer = line[3]
                    if stacklayer not in callstack_lookup:
                        callstack_lookup[stacklayer] = lookup_i
                        lookup_i += 1
                    if line[3] == "main":
                        seen_main = True
                    trace.append(stacklayer)
                max_stack_len = max(max_stack_len, len(trace))
                allocated[l[1]] = {"trace": trace, "size": int(l[2])}
            elif l[0] == 'free':
                if l[1] == '0x0':
                    continue
                record = allocated[l[1]]
                record["usecs"] = int(l[2])
                encoded_trace = list(
                    map(lambda x: str(callstack_lookup[x]), record['trace']))
                encoded_trace += ['0'] * (dim - 1 - len(encoded_trace))
                if (encoded_trace[:7] == ['3', '24', '50', '40', '30', '31', '2']) and (record["usecs"] > 5000): # wash data
                    print(encoded_trace, record['usecs'])
                    continue
                output.write(json.dumps(record))
                output.write('\n')
                encoded_output.write(f'[[{record["size"]},')
                encoded_output.write(','.join(encoded_trace[:dim-1]))
                encoded_output.write(f'],[{record["usecs"]}]]\n')
            else:
                print("read wrong input!!!")
                print(l)
                exit(-1)
    lookup_table_f = open(args.output_pref+'_table.json', 'w')
    json.dump({"table": callstack_lookup, "i": lookup_i}, lookup_table_f)
    lookup_table_f.close()
    output.close()
    encoded_output.close()
    print(f"max stack length: {max_stack_len}")


if __name__ == '__main__':
    main(parse())
