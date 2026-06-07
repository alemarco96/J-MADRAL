import argparse
import json
import operator
import time


def parse_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument("--p_qrels_filename",
                      dest="p_qrels_filename",
                      type=str,
                      required=True)
    args.add_argument("--r_qrels_filename",
                      dest="r_qrels_filename",
                      type=str,
                      required=True)
    args.add_argument("--p_queries_filename",
                      dest="p_queries_filename",
                      type=str,
                      required=True)
    args.add_argument("--r_queries_filename",
                      dest="r_queries_filename",
                      type=str,
                      required=True)
    args.add_argument("--p_crun1_filename",
                      dest="p_crun1_filename",
                      type=str,
                      required=True)
    args.add_argument("--p_crun2_filename",
                      dest="p_crun2_filename",
                      type=str,
                      required=True)
    args.add_argument("--r_crun1_filename",
                      dest="r_crun1_filename",
                      type=str,
                      required=True)
    args.add_argument("--r_crun2_filename",
                      dest="r_crun2_filename",
                      type=str,
                      required=True)
    args.add_argument("--p_output_filename",
                      dest="p_output_filename",
                      type=str,
                      required=True)
    args.add_argument("--r_output_filename",
                      dest="r_output_filename",
                      type=str,
                      required=True)
    args.add_argument("--sort_criteria",
                      dest="sort_criteria",
                      type=str,
                      choices=["merge", "run1", "run2"],
                      default="merge",
                      required=False)
    args.add_argument("--p_min_relevance",
                      dest="p_min_relevance",
                      type=int,
                      default=100,
                      required=False)
    args.add_argument("--r_min_relevance",
                      dest="r_min_relevance",
                      type=int,
                      default=1,
                      required=False)
    args.add_argument("--max_rank_for_exclude",
                      dest="max_rank_for_exclude",
                      type=int,
                      default=0,
                      required=False)
    args.add_argument("--min_rank_for_hardnegatives",
                      dest="min_rank_for_hardnegatives",
                      type=int,
                      default=50,
                      required=False)
    args.add_argument("--max_rank_for_hardnegatives",
                      dest="max_rank_for_hardnegatives",
                      type=int,
                      default=150,
                      required=False)
    args.add_argument("--num_p_hardnegatives",
                      dest="num_p_hardnegatives",
                      type=int,
                      default=100,
                      required=False)
    args.add_argument("--num_r_hardnegatives",
                      dest="num_r_hardnegatives",
                      type=int,
                      default=100,
                      required=False)
    args.add_argument("--min_num_hardnegatives",
                      dest="min_num_hardnegatives",
                      type=int,
                      default=28,
                      required=False)
    args.add_argument("--num_documents",
                      dest="num_documents",
                      type=int,
                      default=1000,
                      required=False)

    return args.parse_args()


def main():
    NS_IN_S = 1000 * 1000 * 1000

    # ------------------------------------------------------------------------------------------------------------------
    # Get the command line parameters passed to the script.
    # ------------------------------------------------------------------------------------------------------------------
    args = parse_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Read the qrels data from disk.
    # ------------------------------------------------------------------------------------------------------------------
    p_qrels_data = {}
    r_qrels_data = {}

    for fn, data, q_cast, d_cast, min_rel in zip([args.p_qrels_filename, args.r_qrels_filename],
                                                 [p_qrels_data, r_qrels_data],
                                                 [True, False], [False, False],
                                                 [args.p_min_relevance, args.r_min_relevance]):
        st = time.time_ns()
        with open(fn, "rt", encoding="utf-8") as fi:
            for query_id, _, doc_id, relevance in map(str.split, fi):
                query_id = int(query_id) if q_cast else str(query_id)
                doc_id = int(doc_id) if d_cast else str(doc_id)
                relevance = int(relevance)

                inner = data.get(query_id, (set(), set()))
                if relevance >= min_rel:
                    inner[0].add(doc_id)
                else:
                    inner[1].add(doc_id)
                data[query_id] = inner
                del inner
            try:
                del query_id, _, doc_id, relevance
            except UnboundLocalError:
                pass
        del fi

        print(f"Read {len(data)} relevance judgements in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
        del st
    try:
        del fn, data, q_cast, d_cast
    except UnboundLocalError:
        pass

    print(f"Found {len(p_qrels_data)} queries with relevance judgements for product search.", flush=False)
    print(f"Found {len(r_qrels_data)} queries with relevance judgements for review search.", flush=False)
    print("", flush=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Read the queries data from disk.
    # ------------------------------------------------------------------------------------------------------------------
    p_queries_data = {}
    r_queries_data = {}

    for fn, data, q_cast in zip([args.p_queries_filename, args.r_queries_filename],
                                [p_queries_data, r_queries_data],
                                [True, False]):
        st = time.time_ns()
        with open(fn, "rt", encoding="utf-8") as fi:
            data.update({int(v["id"]) if q_cast else str(v["id"]): {"text": v["text"], "aspects": v["aspects"]}
                         for v in map(json.loads, fi)})
        del fi

        print(f"Read {len(data)} queries in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
        del st
    try:
        del fn, data, q_cast
    except UnboundLocalError:
        pass

    print(f"Found {len(p_queries_data)} queries with text and aspects for product search.", flush=False)
    print(f"Found {len(r_queries_data)} queries with text and aspects for review search.", flush=False)
    print("", flush=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Read the run data, and extract the important documents.
    # ------------------------------------------------------------------------------------------------------------------
    p_run1_data = {}
    p_run2_data = {}
    r_run1_data = {}
    r_run2_data = {}

    for fn, data, q_cast, d_cast in zip([args.p_crun1_filename, args.p_crun2_filename,
                                         args.r_crun1_filename, args.r_crun2_filename],
                                        [p_run1_data, p_run2_data, r_run1_data, r_run2_data],
                                        [True, True, False, False], [False, False, False, False]):
        curr_data = {}

        st = time.time_ns()
        with open(fn, "rt", encoding="utf-8") as fi:
            for query_id, _, doc_id, rank, _, _ in map(str.split, fi):
                query_id = int(query_id) if q_cast else str(query_id)
                doc_id = int(doc_id) if d_cast else str(doc_id)
                rank = int(rank)

                inner = data.get(query_id, {})
                inner[rank] = doc_id
                curr_data[query_id] = inner
                del inner
            try:
                del query_id, _, doc_id, rank
            except UnboundLocalError:
                pass
        del fi

        data = {k1: [k2 for i2, k2 in sorted(v1.items(), key=operator.itemgetter(0))]
                for k1, v1 in sorted(curr_data.items(), key=operator.itemgetter(0))}
        del curr_data

        print(f"Read {len(data)} run data in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
        del st
    try:
        del fn, data, q_cast, d_cast
    except UnboundLocalError:
        pass

    print(f"Found {len(p_run1_data)} and {len(p_run2_data)} run data for product search.", flush=False)
    print(f"Found {len(r_run1_data)} and {len(r_run2_data)} run data for review search.", flush=False)
    print("", flush=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Process all input data, generate the training data, and write it to disk.
    # ------------------------------------------------------------------------------------------------------------------
    p_queries_id = set(p_queries_data.keys() & p_qrels_data.keys() & (p_run1_data.keys() | p_run2_data.keys()))
    r_queries_id = set(r_queries_data.keys() & r_qrels_data.keys() & (r_run1_data.keys() | r_run2_data.keys()))

    print(f"Found {len(p_queries_id)} valid queries for product search.", flush=False)
    print(f"Found {len(r_queries_id)} valid queries for review search.", flush=False)
    print("", flush=True)

    st = time.time_ns()
    ctr = 0
    for fn, queries_id, task, queries_data, qrels_data, run1_data, run2_data, num_hn in zip(
        [args.p_output_filename, args.r_output_filename],
        [p_queries_id, r_queries_id], ["p", "r"],
        [p_queries_data, r_queries_data],
        [p_qrels_data, r_qrels_data],
        [p_run1_data, r_run1_data],
        [p_run2_data, r_run2_data],
        [args.num_p_hardnegatives, args.num_r_hardnegatives]):

        with open(fn, "wt", encoding="utf-8") as fo:
            for query_id in sorted(queries_id):
                r1_data = run1_data.get(query_id, {})
                r2_data = run2_data.get(query_id, {})
                q_data = qrels_data.get(query_id, (set(), set()))

                # Find the rank of each positive document.
                p_rank = [(k, r1_data.get(k, args.num_documents), r2_data.get(k, args.num_documents))
                          for k in q_data[0]]
                if args.sort_criteria == "merge":
                    p_rank = [(k, min(r1, r2), max(r1, r2)) for k, r1, r2 in p_rank]
                    p_rank = sorted(p_rank, key=operator.itemgetter(1, 2), reverse=False)
                elif args.sort_criteria == "run1":
                    p_rank = sorted(p_rank, key=operator.itemgetter(1, 2), reverse=False)
                elif args.sort_criteria == "run2":
                    p_rank = sorted(p_rank, key=operator.itemgetter(2, 1), reverse=False)
                else:
                    raise ValueError(f"Unrecognised sort criteria: found ¨{args.sort_criteria}.")

                # Find the rank of each negative document.
                n_rank = [(k, r1_data.get(k, args.num_documents), r2_data.get(k, args.num_documents))
                          for k in q_data[1]]
                if args.sort_criteria == "merge":
                    n_rank = [(k, min(r1, r2), max(r1, r2)) for k, r1, r2 in n_rank]
                    n_rank = sorted(n_rank, key=operator.itemgetter(1, 2), reverse=False)
                elif args.sort_criteria == "run1":
                    n_rank = sorted(n_rank, key=operator.itemgetter(1, 2), reverse=False)
                elif args.sort_criteria == "run2":
                    n_rank = sorted(n_rank, key=operator.itemgetter(2, 1), reverse=False)
                else:
                    raise ValueError(f"Unrecognised sort criteria: found ¨{args.sort_criteria}.")

                # Find which documents to exclude, due to being evaluated in the relevance judgements
                # or being ranked highly in the input run data.
                exclude_id = set.union(*q_data)

                # Find the rank of each hard negative document.
                hn_rank = set.union(*[
                    {k for k, i in r1_data.items() if k not in exclude_id and \
                     args.min_rank_for_hardnegatives <= i < args.max_rank_for_hardnegatives},
                    {k for k, i in r2_data.items() if k not in exclude_id and \
                     args.min_rank_for_hardnegatives <= i < args.max_rank_for_hardnegatives}
                ])
                hn_rank = [(k, r1_data.get(k, args.num_documents), r2_data.get(k, args.num_documents))
                           for k in hn_rank]
                if args.sort_criteria == "merge":
                    hn_rank = [(k, min(r1, r2), max(r1, r2)) for k, r1, r2 in hn_rank]
                    hn_rank = sorted(hn_rank, key=operator.itemgetter(1, 2), reverse=False)[:num_hn]
                elif args.sort_criteria == "run1":
                    hn_rank = sorted(hn_rank, key=operator.itemgetter(1, 2), reverse=False)[:num_hn]
                elif args.sort_criteria == "run2":
                    hn_rank = sorted(hn_rank, key=operator.itemgetter(2, 1), reverse=False)[:num_hn]
                else:
                    raise ValueError(f"Unrecognised sort criteria: found ¨{args.sort_criteria}.")

                # Ensure that all training data stored to disk has the sufficient amount of data required for training.
                if len(p_rank) >= 1 and (len(n_rank) + len(hn_rank)) >= args.min_num_hardnegatives:
                    print(json.dumps({
                        "query_id": query_id,
                        "task": task,
                        "p_rank": p_rank,
                        "n_rank": n_rank,
                        "hn_rank": hn_rank
                    }), file=fo, flush=False)
                del r1_data, r2_data, q_data, exclude_id, p_rank, n_rank, hn_rank

                ctr += 1
                if (ctr % 10000) == 0:
                    print(f"Processed {ctr} train instances in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
            try:
                del query_id
            except UnboundLocalError:
                pass

            fo.flush()
        del fo
    try:
        del queries_id, task, queries_data, qrels_data, run1_data, run2_data, num_hn
    except UnboundLocalError:
        pass

    if (ctr % 10000) != 0:
        print(f"Processed {ctr} train instances in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del ctr, st

    print("\nDone!\n", flush=True)


if __name__ == '__main__':
    main()
