import torch
import re
from matplotlib import pyplot as plt

from model_ABFCG1 import *
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
from data_process1 import *
from transformers import get_linear_schedule_with_warmup
import sys
# sys.path.append('../../')
# from BeamSearch import beam_search_generate

# from config import *
# from Embedding4ast import *
# from evaluator.CodeBLEU.calc_code_bleu import get_codebleu

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
def run_train(
    dataloader,
    num_epoch,
    model,
    optimizer,
    scheduler,
    save_weight_file,
    load_weight_file = None,
    early_stop = False,
    dataloader_valid=None,
    dataloader_test=None,
    acumulate_grad_num = None,
    accum_iter=1,
    patience=5
):

    criterion = nn.CrossEntropyLoss(ignore_index=0)


    print(f"train on {device} under the lr {optimizer.param_groups[0]['lr']}")

    total_tokens = 0
    total_loss = 0
    iter_num = 0
    n_accum = 0


    stop = 0
    loss_list = []
    ppl_list = []
    best_test_bleu_em_partly = 0.0
    best_test_bleu_em = 0.0
    best_valid_bleu_em = 0.0
    best_valid_ppl = 0.0
    print(f"Plan to train {num_epoch} epoch. Start running now ...")
    f = open(save_weight_file.split(".")[0]+"_train_log.txt", "w")
    if load_weight_file!=None:
        model.load_state_dict(
            torch.load(load_weight_file, map_location=device))
        print(f"loading from {load_weight_file}\n")
        f.write(f"loading from {load_weight_file}\n")

    f.write("epoch " + "\t" + "loss " + "\t" + "fusion-loss " + "\t" + "valid-bleu " + "\t"+ "valid-em " + "\t"+ "lr " +"\n")
    f.close()
    for epoch in range(1, num_epoch + 1):
        torch.cuda.empty_cache()
        model.to(device)
        model.train()

        if stop>patience and early_stop:
            print(f"early_stop after {epoch-1} epoch")
            break

        ntoken = 0


        iter_num = 0
        total_loss = 0.0
        total_fusion_loss = 0.0
        sample_cnt = 0
        optimizer.zero_grad(set_to_none=True)
        for batch in tqdm(dataloader,desc=f"train in {epoch} epoch"):

            nl_tensors, code_tensors, related_code_tensors,related_nl_tensors,related_ast_tokens,related_ast_matrice = batch
                # related_cfg_tensors
            sample_cnt += len(nl_tensors)*3

            for codetoken_tensor in code_tensors:
                ntoken += len([token for token in codetoken_tensor if token!=0])
            # nl_tensors = torch.roll(nl_tensors, shifts=-1, dims=1)
            # nl_tensors[:, -1] = 0
            nl_tensors = torch.tensor(nl_tensors, dtype=torch.long).to(device)
            code_tensors = torch.tensor(code_tensors, dtype=torch.long).to(device)
            related_code_tensors = torch.tensor(related_code_tensors,dtype=torch.long).to(device)
            related_nl_tensors = torch.tensor(related_nl_tensors,dtype=torch.long).to(device)
            related_ast_tokens = torch.tensor(related_ast_tokens,dtype=torch.long).to(device)
            related_ast_matrice = torch.tensor(related_ast_matrice,dtype=torch.long).to(device)


            # fusion_loss
            out,fusion_loss = model(
                nl_tensors,
                code_tensors,
                related_code_tensors,
                related_nl_tensors,
                related_ast_tokens,related_ast_matrice
            )
            code_tensors = torch.roll(code_tensors, shifts=-1, dims=1)
            code_tensors[:, -1] = 0
            loss = criterion(out.reshape(-1, out.size(-1)).contiguous(),
                             code_tensors.reshape(-1).contiguous())

            total_fusion_loss += fusion_loss.item()
            loss+=fusion_loss
            loss.backward()

            total_loss += loss.item()
            # total_loss += fusion_loss.item()

            if acumulate_grad_num is not None and iter_num%acumulate_grad_num==0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

                n_accum += 1

            total_tokens += ntoken
            ntoken = 0
            del loss
            iter_num += 1
        loss_list.append(total_loss / iter_num)
        print(f"Epoch {epoch} average loss: {total_loss / iter_num} average fusion loss:{total_fusion_loss/iter_num}")

        valid_bleu,valid_em = run_test("greedy", dataloader_valid, model, None, partly=True)
        if (len(loss_list) > 2 and loss_list[-1] > min(loss_list[:-1]) ):
            stop += 1
            # optimizer.param_groups[0]['lr']*=0.95
        else:
            loss_list.append(total_loss / iter_num)
            stop = 0
        if best_valid_bleu_em<valid_bleu+valid_em:
            best_valid_bleu_em = valid_bleu+valid_em
            torch.save(model.state_dict(), save_weight_file.split(".")[0] + "_best_valid.pt")
            print(f"saving the weight file to {save_weight_file.split('.')[0]}_best_valid.pt")
        if epoch>num_epoch*0.9:
            test_bleu, test_em = run_test("greedy",dataloader_test,model,None,partly=False)
            if test_bleu+test_em>best_test_bleu_em:
                best_test_bleu_em = test_bleu+test_em
                torch.save(model.state_dict(), save_weight_file.split(".")[0] + "_test.pt")
                print(f"test_bleu:{test_bleu} | test_em:{test_em} | \nsaving the weight file to {save_weight_file.split('.')[0]}_test.pt")
                os.rename("test_output","test_output_best")
                os.rename("test_gold","test_gold_best")
        torch.save(model.state_dict(), save_weight_file.split(".")[0]+"_last.pt")
        print(f" saving the weight file to {save_weight_file.split('.')[0]}_last.pt in {epoch} epoch\n")
        print(f"best valid bleu+em:{best_valid_bleu_em} | best test bleu+em:{best_test_bleu_em} | best test bleu+em:{best_valid_bleu_em}")
        f = open(save_weight_file.split(".")[0] + "_train_log.txt", "a")

        f.write(str(epoch)+"\t"+str(total_loss / iter_num)+"\t"+str(total_fusion_loss / iter_num)+ "\t"+ str(valid_bleu) + "\t"+ str(valid_em)+ "\t"+str(optimizer.param_groups[0]["lr"])+"\n")
        f.close()



def run_test(mode,dataloader,model,
              # nl_vocab_stoi,code_tokens_vocab_stoi, code_tokens_vocab_itos,
             load_weight_file=None,partly=True
              ):
    model.eval()
    model.to(device)
    sample_cnt = 0
    test_cnt = 0
    total_bleu = 0.0
    total_codebleu = 0.0
    em_cnt = 0
    if load_weight_file is not None:
        model.load_state_dict(torch.load(load_weight_file,map_location=device))
    assert mode in ["greedy","beam search"]
    f = open("pred_code.txt", "w")
    if partly:
        f1 = open("valid_output", "w")
        f2 = open("valid_gold", "w")
    else:
        f1 = open("test_output", "w")
        f2 = open("test_gold", "w")
    if mode == "greedy":
        # fp = open("Experiment analysist/result_csnjava_em","w")

        for batch in tqdm(dataloader,desc=f"reasoning"):
            if test_cnt>200 and partly is True:
                break
            nl_tensors, code_tensors, related_code_tensors, related_nl_tensors,related_ast_tokens,related_ast_matrice = batch
            # nl_tensors = torch.roll(nl_tensors, shifts=-1, dims=1)
            # nl_tensors[:, -1] = 0
            nl_tensors = torch.tensor(nl_tensors, dtype=torch.long).to(device)
            code_tensors = torch.tensor(code_tensors, dtype=torch.long).to(device)
            related_code_tensors = torch.tensor(related_code_tensors,dtype=torch.long).to(device)
            related_nl_tensors = torch.tensor(related_nl_tensors,dtype=torch.long).to(device)
            related_ast_tokens = torch.tensor(related_ast_tokens, dtype=torch.long).to(
                device)
            related_ast_matrice = torch.tensor(related_ast_matrice, dtype=torch.long).to(
                device)
            pred_ids = greedy_decode(model, nl_tensors, related_code_tensors, related_nl_tensors,
                                     related_ast_tokens,
                                     related_ast_matrice, 128,
                                         BOS, EOS
                                         )
            test_cnt += nl_tensors.size(0)
            for i in range(nl_tensors.size(0)):
                pred_code_ids = pred_ids.tolist()[i]
                if EOS in pred_code_ids:
                    pred_code_ids = pred_code_ids[:pred_code_ids.index(EOS)]
                pred_code = []
                for ids in pred_code_ids:
                    if str(ids) not in itos.keys():
                        pred_code.append("<unk>")
                    else:
                        pred_code.append(itos[str(ids)])
                pred_code = ' '.join(pred_code).replace("\n","").replace("\r","").replace("\t","").replace("\\","")
                target_code = []
                if EOS in code_tensors[i].tolist():
                    for ids in code_tensors[i][:code_tensors[i].tolist().index(EOS)].tolist():
                        if str(ids) not in itos.keys():
                            target_code.append("<unk>")
                        else:
                            target_code.append(itos[str(ids)])
                else:
                    for ids in code_tensors[i][:].tolist():
                        if str(ids) not in itos.keys():
                            target_code.append("<unk>")
                        else:
                            target_code.append(itos[str(ids)])
                target_code = ' '.join(target_code).replace("\n","").replace("\r","").replace("\t","").replace("\\","")
                f1.write(pred_code + '\n')
                f2.write(target_code + '\n')
                bleu_score = nltk.translate.bleu_score.sentence_bleu([target_code], pred_code,weights=[0,0,0,1])
                total_bleu += bleu_score
                # nl = []
                # for ids in nl_tensors[i][:nl_tensors[i].tolist().index(EOS)].tolist():
                #     if str(ids) not in nl_vocab_itos.keys():
                #         nl.append("<unk>")
                #     else:
                #         nl.append(nl_vocab_itos[str(ids)])

                if target_code==pred_code:
                    em_cnt+=1

        f.close()
        print(f"Average bleu-4: {total_bleu / test_cnt} |Exact Match: {em_cnt/test_cnt}| Total samples: {test_cnt}")
        return total_bleu / test_cnt,em_cnt/test_cnt

    else:
        for batch in tqdm(dataloader,desc="beam searching"):
            # 每个批次的数据
            nl_tensors, code_tensors, related_code_tensors, related_nl_tensors,related_code_ast_nodes_tensors,related_code_ast_edges_tensors,ground_truth_idx = batch
            test_cnt += nl_tensors.size(0)

            nl_tensors = torch.roll(nl_tensors, shifts=-1, dims=1)
            nl_tensors[:, -1] = 0
            nl_tensors = torch.tensor(nl_tensors, dtype=torch.long).to(device)
            codetoken_tensors = torch.tensor(code_tensors, dtype=torch.long).to(device)
            related_code_tensors = torch.tensor(related_code_tensors,dtype=torch.long).to(device)
            related_nl_tensors = torch.tensor(related_nl_tensors,dtype=torch.long).to(device)
            related_code_ast_nodes_tensors = torch.tensor(related_code_ast_nodes_tensors, dtype=torch.long).to(
                device)
            related_code_ast_edges_tensors = torch.tensor(related_code_ast_edges_tensors, dtype=torch.long).to(
                device)
            memory = model.encode(nl_tensors, related_code_tensors, related_nl_tensors, related_code_ast_nodes_tensors, related_code_ast_edges_tensors)
            pred_ids = beam_search_generate(memory,model,len(code_tokens_vocab_stoi),batch_size=memory.size(0),bos_token_id=BOS,pad_token_id=PAD,eos_token_id=EOS,num_beams=1)
            for i in range(nl_tensors.size(0)):
                pred_code_ids = pred_ids.tolist()[i]
                if EOS in pred_code_ids:
                    pred_code_ids = pred_code_ids[:pred_code_ids.index(EOS)]
                pred_code = []
                for ids in pred_code_ids:
                    if str(ids) not in itos.keys():
                        pred_code.append("<unk>")
                    else:
                        pred_code.append(itos[str(ids)])
                pred_code = ' '.join(pred_code)
                target_code = []
                for ids in code_tensors[i][:code_tensors[i].tolist().index(EOS)].tolist():
                    if str(ids) not in itos.keys():
                        target_code.append("<unk>")
                    else:
                        target_code.append(itos[str(ids)])
                target_code = ' '.join(target_code)
                f1.write(pred_code + '\n')
                f2.write(target_code + '\n')
                bleu_score = nltk.translate.bleu_score.sentence_bleu([target_code], pred_code, weights=[0, 0, 0, 1])
                total_bleu += bleu_score
                nl = []
                for ids in nl_tensors[i][:nl_tensors[i].tolist().index(EOS)].tolist():
                    if str(ids) not in nl_vocab_itos.keys():
                        nl.append("<unk>")
                    else:
                        nl.append(nl_vocab_itos[str(ids)])

                if target_code == pred_code:
                    em_cnt += 1

        f.close()
        print(f"Average bleu-4: {total_bleu / test_cnt} |Exact Match: {em_cnt / test_cnt}| Total samples: {test_cnt}")
        return total_bleu / test_cnt, em_cnt / test_cnt
# 这个代码其实和最开始的inference_test()是一样的
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

    batch = Batch4ABFCG("dataset/CSN-JAVA/csn-java.json","codesearchnet", "bolt://localhost:7687", auth=("neo4j", "password"),
                          )

    dataloader = batch.get_dataloader("processed_datasets/CSN-JAVA/csn-java-train_samples_ids.json",16,True)
    dataloader_valid = batch.get_dataloader("processed_datasets/CSN-JAVA/csn-java-valid_samples_ids.json",16,False)
    dataloader_test = batch.get_dataloader("processed_datasets/CSN-JAVA/csn-java-test_samples_ids.json",16,False)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model = EncoderDecoder(
        len(nl_vocab_stoi),len(code_tokens_vocab_stoi),N=6).to(device)


    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-5, betas=(0.9, 0.98), eps=1e-9
    )
    num_epochs =40
    scheduler = get_linear_schedule_with_warmup(optimizer,3000,num_epochs*len(dataloader))
    print("Start")
    run_train(dataloader,num_epochs,model,optimizer,scheduler,
              # load_weight_file="v1_last.pt",
    save_weight_file="v1.pt",early_stop=True,
              acumulate_grad_num=1,dataloader_valid=dataloader_valid,dataloader_test=dataloader_test,patience=8)









