import time

import torch
import re
from matplotlib import pyplot as plt

from model_ABFCG1 import *
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
from data_process1 import *
from transformers import get_linear_schedule_with_warmup
import sys
import argparse

from BeamSearch import beam_search_generate

# from config import *
# from Embedding4ast import *
# from evaluator.CodeBLEU.calc_code_bleu import get_codebleu

BOS = code_tokens_vocab_stoi["<bos>"]
EOS = code_tokens_vocab_stoi["<eos>"]
PAD = code_tokens_vocab_stoi["<pad>"]
UNK = code_tokens_vocab_stoi["<unk>"]
stoi = code_tokens_vocab_stoi
itos = code_tokens_vocab_itos

def codetokens2strings(code_tokens):
    code_str = ''

    def _is_whitespace_token(token):
        return re.fullmatch(r'\s*', token) is not None
    for i in range(1,len(code_tokens)-1):
        special_tokens = ["{","}","[","]","(",")",":",".",",","\n"]
        if code_tokens[i] not in special_tokens and _is_whitespace_token(code_tokens[i]) is not True and i!=1:
            if i>1 and code_tokens[i-1] in special_tokens:
                code_str += code_tokens[i]
            else:
                code_str += " " + code_tokens[i]
        else:
            code_str += code_tokens[i]
    return code_str



def run_test(dataloader, model, load_weight_file=None):
    model.eval()
    model.to(device)

    test_cnt = 0
    total_bleu = 0.0
    em_cnt = 0

    if load_weight_file is not None:
        model.load_state_dict(torch.load(load_weight_file, map_location=device))

    f_pred_log = open("pred_code.txt", "w")
    f_output = open("output.txt", "w")
    f_gold = open("gold.txt", "w")

    start_time = time.time()

    for batch in tqdm(dataloader, desc="greedy decoding"):
        nl_tensors, code_tensors, related_code_tensors, related_nl_tensors, related_ast_tokens, related_ast_matrice = batch

        nl_tensors = torch.tensor(nl_tensors, dtype=torch.long).to(device)
        code_tensors = torch.tensor(code_tensors, dtype=torch.long).to(device)
        related_code_tensors = torch.tensor(related_code_tensors, dtype=torch.long).to(device)
        related_nl_tensors = torch.tensor(related_nl_tensors, dtype=torch.long).to(device)
        related_ast_tokens = torch.tensor(related_ast_tokens, dtype=torch.long).to(device)
        related_ast_matrice = torch.tensor(related_ast_matrice, dtype=torch.long).to(device)

        pred_ids = greedy_decode(model, nl_tensors, related_code_tensors, related_nl_tensors,
                                 related_ast_tokens,
                                 related_ast_matrice, 128,
                                 BOS, EOS
                                 )

        batch_size = nl_tensors.size(0)
        test_cnt += batch_size

        for i in range(batch_size):
            pred_code_ids = pred_ids[i].tolist()
            if EOS in pred_code_ids:
                pred_code_ids = pred_code_ids[:pred_code_ids.index(EOS)]

            pred_code = [itos.get(str(idx), "<unk>") for idx in pred_code_ids]
            pred_code_str = ' '.join(pred_code).replace("\n", "").replace("\r", "").replace("\t", "").replace("\\", "")

            target_ids = code_tensors[i].tolist()
            if EOS in target_ids:
                target_ids = target_ids[:target_ids.index(EOS)]
            target_code = [itos.get(str(idx), "<unk>") for idx in target_ids]
            target_code_str = ' '.join(target_code).replace("\n", "").replace("\r", "").replace("\t", "").replace("\\", "")

            f_output.write(pred_code_str + '\n')
            f_gold.write(target_code_str + '\n')
            f_pred_log.write(pred_code_str + '\n')

            bleu_score = nltk.translate.bleu_score.sentence_bleu([target_code_str], pred_code_str, weights=[0, 0, 0, 1])
            total_bleu += bleu_score

            if pred_code_str == target_code_str:
                em_cnt += 1

    f_output.close()
    f_gold.close()
    f_pred_log.close()

    time_cost = time.time() - start_time
    print(f"Average BLEU-4: {total_bleu / test_cnt:.4f} | Exact Match: {em_cnt / test_cnt:.4f} | "
          f"Total Inference Time: {time_cost:.2f}s | Total Samples: {test_cnt}")

    return total_bleu / test_cnt, em_cnt / test_cnt



def run_train(
    dataloader,
    num_epoch,
    model,
    optimizer,
    scheduler,
    dataset_name,
    save_weight_file,
    load_weight_file=None,
    early_stop=False,
    dataloader_valid=None,
    acumulate_grad_num=None,
    patience=5
):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    print(f"Train on {device} under lr {optimizer.param_groups[0]['lr']}")

    best_valid_score = 0.0
    no_improve_count = 0
    loss_list = []

    f = open(save_weight_file.split(".")[0] + "_train_log.txt", "w")
    if load_weight_file:
        model.load_state_dict(torch.load(load_weight_file, map_location=device))
        print(f"Loaded weights from {load_weight_file}")
        f.write(f"Loaded weights from {load_weight_file}\n")

    f.write("epoch\tloss\tfusion-loss\tvalid-bleu\tvalid-em\tlr\n")
    f.close()

    for epoch in range(1, num_epoch + 1):
        torch.cuda.empty_cache()
        model.to(device)
        model.train()

        if early_stop and no_improve_count >= patience:
            print(f"Early stopping triggered at epoch {epoch - 1}")
            break

        total_loss = 0.0
        total_fusion_loss = 0.0
        iter_num = 0
        optimizer.zero_grad(set_to_none=True)

        for batch in tqdm(dataloader, desc=f"Training epoch {epoch}"):
            nl_tensors, code_tensors, related_code_tensors, related_nl_tensors, related_ast_tokens, related_ast_matrice = batch

            nl_tensors = torch.tensor(nl_tensors, dtype=torch.long).to(device)
            code_tensors = torch.tensor(code_tensors, dtype=torch.long).to(device)
            related_code_tensors = torch.tensor(related_code_tensors, dtype=torch.long).to(device)
            related_nl_tensors = torch.tensor(related_nl_tensors, dtype=torch.long).to(device)
            related_ast_tokens = torch.tensor(related_ast_tokens, dtype=torch.long).to(device)
            related_ast_matrice = torch.tensor(related_ast_matrice, dtype=torch.long).to(device)

            output, fusion_loss = model(
                nl_tensors,
                code_tensors,
                related_code_tensors,
                related_nl_tensors,
                related_ast_tokens,
                related_ast_matrice
            )

            code_shifted = torch.roll(code_tensors, shifts=-1, dims=1)
            code_shifted[:, -1] = 0

            loss = criterion(output.reshape(-1, output.size(-1)), code_shifted.reshape(-1))
            loss += fusion_loss

            loss.backward()
            total_loss += loss.item()
            total_fusion_loss += fusion_loss.item()
            iter_num += 1

            if acumulate_grad_num is not None and iter_num % acumulate_grad_num == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            del loss

        avg_loss = total_loss / iter_num
        avg_fusion_loss = total_fusion_loss / iter_num
        loss_list.append(avg_loss)

        print(f"Epoch {epoch} - avg loss: {avg_loss:.4f}, fusion loss: {avg_fusion_loss:.4f}")

        valid_bleu, valid_em = run_test("greedy", dataloader_valid, model)
        current_score = valid_bleu + valid_em

        if current_score > best_valid_score:
            best_valid_score = current_score
            no_improve_count = 0
            torch.save(model.state_dict(), save_weight_file.split(".")[0] + f"_best_valid_{dataset_name}.pt")
            print(f"Saved best weights to {save_weight_file.split('.')[0]}_best_valid_{dataset_name}.pt")
        else:
            no_improve_count += 1
            print(f"No improvement. Patience counter: {no_improve_count}/{patience}")

        torch.save(model.state_dict(), save_weight_file.split(".")[0] + f"_last_{dataset_name}.pt")

        f = open(save_weight_file.split(".")[0] + "_train_log.txt", "a")
        f.write(f"{epoch}\t{avg_loss:.4f}\t{avg_fusion_loss:.4f}\t{valid_bleu:.4f}\t{valid_em:.4f}\t{optimizer.param_groups[0]['lr']:.6f}\n")
        f.close()


def greedy_decode(model, nl, related_code, related_nl, related_ast_tokens, related_ast_matrice, max_len, start_symbol,
                  end_symbol):

    memory = model.encode(nl, related_code, related_nl,related_ast_tokens, related_ast_matrice)
    ys = torch.ones(nl.size(0), 1).fill_(start_symbol).type_as(nl.data)

    done = [False] * nl.size(0)


    for i in range(max_len - 1):
        if all(done):
            break

        prob = model.decode(
            memory, None, ys, None
        )

        prob = nn.Softmax(dim=-1)(prob[:, -1])

        _, next_word = torch.max(prob, dim=-1)

        ys = torch.cat(
            [ys, next_word.data.view(nl.size(0), 1)], dim=-1
        )

        for batch_idx in range(nl.size(0)):
            if not done[batch_idx]:
                if next_word[batch_idx] == end_symbol:
                    done[batch_idx] = True
    return ys


#





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Path to dataset JSON')
    parser.add_argument('--source_name', type=str, required=True, help='Source name (e.g., deepcom)')
    parser.add_argument('--address', type=str, required=True, help='Neo4j address (e.g., bolt://localhost:7687)')
    parser.add_argument('--username', type=str, required=True, help='Neo4j username')
    parser.add_argument('--password', type=str, required=True, help='Neo4j password')
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--valid_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, required=True)
    parser.add_argument('--load_weight', type=str, default=None)
    parser.add_argument('--save_weight', type=str, required=True)

    args = parser.parse_args()

    batch = Batch4ABFCG(args.dataset, args.source_name, args.address, auth=(args.username, args.password))
    dataloader = batch.get_dataloader(args.train_path, 12, True)
    dataloader_valid = batch.get_dataloader(args.valid_path, 16, False)
    dataloader_test = batch.get_dataloader(args.test_path, 16, False)

    model = EncoderDecoder(len(nl_vocab_stoi), len(code_tokens_vocab_stoi), N=6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-9)

    num_epochs = 40
    torch.manual_seed(1234)
    scheduler = get_linear_schedule_with_warmup(optimizer, 3000, num_epochs * len(dataloader))

    print("Start")
    run_train(dataloader, num_epochs, model, optimizer, scheduler, dataset_name=args.source_name,
              load_weight_file=args.load_weight,
              save_weight_file=args.save_weight,
              early_stop=True,
              acumulate_grad_num=1,
              dataloader_valid=dataloader_valid,
              patience=8)

    run_test(dataloader_test, model, args.save_weight.replace('.pt', '_test.pt'))







