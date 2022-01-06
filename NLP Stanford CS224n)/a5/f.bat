:: Pretrain the model
C:\ProgramData\Anaconda3\python.exe src/run.py pretrain vanilla wiki.txt ^
    --writing_params_path vanilla.pretrain.params
:: Finetune the model
C:\ProgramData\Anaconda3\python.exe src/run.py finetune vanilla wiki.txt ^
    --reading_params_path vanilla.pretrain.params ^
    --writing_params_path vanilla.finetune.params ^
    --finetune_corpus_path birth_places_train.tsv
:: Evaluate on the dev set; write to disk
C:\ProgramData\Anaconda3\python.exe src/run.py evaluate vanilla wiki.txt ^
    --reading_params_path vanilla.finetune.params ^
    --eval_corpus_path birth_dev.tsv ^
    --outputs_path vanilla.pretrain.dev.predictions
:: Evaluate on the test set; write to disk
C:\ProgramData\Anaconda3\python.exe src/run.py evaluate vanilla wiki.txt ^
    --reading_params_path vanilla.finetune.params ^
    --eval_corpus_path birth_test_inputs.tsv ^
    --outputs_path vanilla.pretrain.test.predictions

pause