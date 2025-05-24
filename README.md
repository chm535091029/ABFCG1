# Align Before Fusion and Code Generation
we propose a novel model that utilizes the code knowledge graph to generate source code, which can establish connections between related programs based on their text similarity and represent the program with multiple representations.
Our model is named ABFCG, which first Aligns features of different representations Before Fusion and Code Generation.

Our method mainly including two stages:(1) CodeKG establishment (2) Generating code by ABFCG.

Main dependency is depicted in 'requirement.txt'.
## CodeKG establishment
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
##  Generating code by ABFCG
To train and then test our model, run the script below in main folder
```
run.sh
```
Notice that the vocab in the top of model_ABFCG1.py should match your training dataset.

To further access all the four metrics, run
```
cal_codebleu_rouge.py
```