# Align Before Fusion and Code Generation
we propose a novel model that utilizes the code knowledge graph to generate source code, which can establish connections between related programs based on their text similarity and represent the program with multiple representations.
Our model is named ABFCG, which first Aligns features of different representations Before Fusion and Code Generation.


## Main Dependency
```
javalang                      0.13.0
py2neo                        2021.2.3
rouge                         1.0.1
sklearn                       0.0.post4
tokenizers                    0.13.3
tomli                         2.0.1
torch                         1.13.1+cu116
torch-geometric               2.3.1
torchaudio                    0.13.1+cu116
torchdata                     0.6.1
torchtext                     0.14.0
torchvision                   0.14.1+cu116
tree-sitter                   0.20.2

```
Our method mainly including two stages:(1) CodeKG establishment (2) Generating code by ABFCG.
## CodeKG establishment
We have published the processed datasets in the dir named "processed_datasets", if you want to create your dataset, keep reading this section, or you can skip it.
### Create KG 
We use Neo4j to store the knowledge graph, so please install it before this step. And then create a knowledge graph with your own address, username and password. 
The function to establish CodeKG is in the folder of
```
ABFCG/CodeKG/create_kg.py
```
The usage of this method is below
```
add_NL_PL_java_concode("your_address",auth=("user_name", "password"),filename="your original dataset path")
add_nl_similarity_relations("your_address",auth=("user_name", "password"),filename="your original dataset path")
```
Due to the different schemes and fields between different datasets, maybe you need to change a little bit fields when these functions read and parse data.
### Export and preprocess data for training
After establish the CodeKG, we need to export all the examples with text-smilar relation.
In the main folder,
```
data_process1.py
```
#### Automatically process dataset

To be convenient, we set a method in the class "Batch4ABFCG" named "create_dataset", which can create the method automatically.
Here we give an example
```
batch = Batch4ABFCG("csn-java.json","codesearchnet","bolt://localhost:7687",auth=("neo4j", "password"))
batch.create_dataset(32,128,92,32,4,2)
```
You will acquire following files 
```angular2html
csn-java_code_tokens_vocab.json
csn-java_nl_vocab.json
csn-java-nontrain_samples_guid.txt
csn-java-test_samples.json
csn-java-test_samples_ids.json
csn-java-train_samples.json
csn-java-train_samples_guid.txt
csn-java-train_samples_ids.json
csn-java-valid_samples.json
csn-java-valid_samples_ids.json
codesearchnet-nontrain_att
codesearchnet-train_att
```
where "codesearchnet-nontrain_att" and "codesearchnet-train_att" are the AST Adjacency matrix.
You can directly obtain the data in link：https://drive.google.com/drive/folders/19zvi_xbGHwdcXGTU2R8-yWhz7WusMNHq?usp=drive_link
##  Generating code by ABFCG
To train our model, run the python file below in main folder
```
run_ABFCG1.py
```
The function "run_train()" implement the process of training (our model's hyper-parameters have been set already, you don't need to change them if you want to reproduce it.)
```
    run_train(dataloader,num_epochs,model,optimizer,scheduler,
              # load_weight_file="v1_last.pt",
    save_weight_file="v1.pt",early_stop=True,
              acumulate_grad_num=1,dataloader_valid=dataloader_valid,dataloader_test=dataloader_test,patience=8)
```
Notice that the vocab in the top of model_ABFCG1.py should match your training dataset.
After training it will automatically test and create the generated code and ground truth
```
test_output
test_gold
```
To further access all the four metrics, run
```
cal_codebleu_rouge.py
```