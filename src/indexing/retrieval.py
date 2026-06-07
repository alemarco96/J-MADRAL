import argparse
import io
import numpy
import os
import time
import torch


def parse_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument("--query_index_folder",
                      dest="query_index_folder",
                      type=str,
                      required=True)
    args.add_argument("--doc_index_folder",
                      dest="doc_index_folder",
                      type=str,
                      required=True)
    args.add_argument("--run_filename",
                      dest="run_filename",
                      type=str,
                      required=True)
    args.add_argument("--top_k",
                      dest="top_k",
                      type=int,
                      default=1000,
                      required=False)
    args.add_argument("--chunk_size",
                      dest="chunk_size",
                      type=int,
                      default=10000,
                      required=False)

    return args.parse_args()


def load_faiss_header(file: io.BufferedReader) -> tuple[int, int]:
    file.seek(4, io.SEEK_SET)
    emb_size = int.from_bytes(file.read(4), byteorder="little", signed=False)
    num_vectors = int.from_bytes(file.read(8), byteorder="little", signed=False)
    file.seek(45, io.SEEK_SET)

    return emb_size, num_vectors


def main():
    NS_IN_S = 1000 * 1000 * 1000

    # ------------------------------------------------------------------------------------------------------------------
    # Get the command line parameters passed to the script.
    # ------------------------------------------------------------------------------------------------------------------
    args = parse_args()

    # ------------------------------------------------------------------------------------------------------------------
    # Load the queries embedding data.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()

    queries_embedding = []
    with open(os.path.join(args.query_index_folder, "index"), "rb") as fi:
        # Read FAISS index header.
        emb_size, num_vectors = load_faiss_header(fi)

        # Read all the embeddings.
        while True:
            embeddings = fi.read(args.chunk_size * emb_size * 4)
            if not embeddings:
                del embeddings
                break

            embeddings = numpy.frombuffer(embeddings, dtype=numpy.float32)
            embeddings = torch.from_numpy(embeddings).to(dtype=torch.float32)
            embeddings = embeddings.view(-1, emb_size)

            queries_embedding.append(embeddings)
            del embeddings
        del emb_size, num_vectors
    del fi

    queries_embedding = torch.cat(queries_embedding, dim=0)

    # ------------------------------------------------------------------------------------------------------------------
    # Load the queries id data.
    # ------------------------------------------------------------------------------------------------------------------
    queries_id = []
    with open(os.path.join(args.query_index_folder, "docid"), "rt", encoding="utf-8") as fi:
        for line in fi:
            if line == "":
                continue

            queries_id.append(line.strip())
        try:
            del line
        except UnboundLocalError:
            pass
    del fi

    if len(queries_id) != queries_embedding.shape[0]:
        raise ValueError(f"Invalid index: mismatch in the number of queries "
                         f"(found {len(queries_id)} and {queries_embedding.shape[0]}).")

    print(f"Query index loaded in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del st

    # ------------------------------------------------------------------------------------------------------------------
    # Load and process the document index in small pieces.
    # ------------------------------------------------------------------------------------------------------------------
    best_relevance = torch.tensor([], dtype=torch.float32)
    best_index = torch.tensor([], dtype=torch.int64)

    st = time.time_ns()
    with open(os.path.join(args.doc_index_folder, "index"), "rb") as fi:
        # Read FAISS index header.
        emb_size, num_vectors = load_faiss_header(fi)

        last = 0
        ctr = 0
        while True:
            # Read the next chunk of documents embedding.
            docs_embedding = fi.read(args.chunk_size * emb_size * 4)
            if not docs_embedding:
                del docs_embedding
                break

            docs_embedding = numpy.frombuffer(docs_embedding, dtype=numpy.float32)
            docs_embedding = torch.from_numpy(docs_embedding).to(dtype=torch.float32)
            docs_embedding = docs_embedding.view(-1, emb_size)
            num_docs = docs_embedding.shape[0]

            # Compute the queries-documents similarity.
            qd_similarity = torch.matmul(queries_embedding, docs_embedding.transpose(0, 1))
            del docs_embedding

            # Extract the top-k most relevant documents in the current chunk.
            top_relevance, top_index = torch.topk(qd_similarity, k=args.top_k, dim=1, largest=True, sorted=True)
            top_index += ctr
            del qd_similarity

            # Concatenate the best with current top-k.
            top_relevance = torch.cat([best_relevance, top_relevance], dim=1)
            top_index = torch.cat([best_index, top_index], dim=1)

            # Retain the new best top-k elements.
            best_relevance, curr_index = torch.topk(top_relevance, k=args.top_k, dim=1, largest=True, sorted=True)
            best_index = torch.gather(top_index, 1, curr_index)
            del top_relevance, top_index

            # Increase the counter.
            ctr += num_docs
            if (ctr - last) >= 100000:
                print(f"Processed {ctr} / {num_vectors} = {ctr * 100 / num_vectors:.2f}% "
                      f"documents in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
                last = ctr
            del num_docs
        del emb_size
    del fi
    del queries_embedding

    if (ctr - last) >= 10000:
        print(f"Processed {ctr} / {num_vectors} = {ctr * 100 / num_vectors:.2f}% "
              f"documents in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del last

    if ctr != num_vectors:
        raise ValueError(f"Invalid index: mismatch in the number of documents "
                         f"(count: {ctr}, index: {num_vectors}).")
    del ctr

    print(f"Matching completed in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del st

    # ------------------------------------------------------------------------------------------------------------------
    # Load and process the document id list.
    # ------------------------------------------------------------------------------------------------------------------
    st = time.time_ns()
    with open(os.path.join(args.doc_index_folder, "docid"), "rt", encoding="utf-8") as fi:
        documents_id = [line[:-1] for line in fi]
    del fi

    if len(documents_id) != num_vectors:
        raise ValueError(f"Invalid docid: mismatch in the number of documents "
                         f"(found: {len(documents_id)}, index: {num_vectors}).")
    del num_vectors

    best_id = [[documents_id[v2] for v2 in v1] for v1 in best_index.tolist()]
    del documents_id

    print(f"Document id matching completed in {(time.time_ns() - st) / NS_IN_S:.3f} s.", flush=True)
    del st

    # ------------------------------------------------------------------------------------------------------------------
    # Write the final run to disk.
    # ------------------------------------------------------------------------------------------------------------------
    with open(args.run_filename, "wt", encoding="utf-8") as fo:
        for i1 in range(len(queries_id)):
            query_id = queries_id[i1]
            documents_relevance = best_relevance[i1].tolist()
            documents_id = best_id[i1]

            for rank, v1 in enumerate(zip(documents_relevance, documents_id)):
                doc_relevance, doc_id = v1

                print(f"{query_id}\tQ0\t{doc_id}\t{rank}\t{doc_relevance:.6f}\tx", file=fo, flush=False)
                del doc_relevance, doc_id
            try:
                del rank, v1
            except UnboundLocalError:
                pass

            del query_id, documents_relevance, documents_id
        try:
            del i1
        except UnboundLocalError:
            pass

        fo.flush()
    del fo

    print("\nDone!\n", flush=True)


if __name__ == '__main__':
    main()
