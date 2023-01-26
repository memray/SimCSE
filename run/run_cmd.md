fuser -v /dev/nvidia* | awk '{ print $0 }' | xargs -n1 kill -9

## cc_v2
cd /export/home/project/search/uir_best_cc
sh run/cc_v2/gpu16/wiki20+cc80.moco2e14.hybrid_rc20gen80.bs2048.gpu8.sh  # pod/sfr-pod-ruimeng.a100-16-1, no OOM, 237G/1.31T
sh run/cc_v2/gpu16/wiki20+cc80.moco2e14.hybrid_rc20gen80.bs4096.gpu16.sh  # pod/sfr-pod-ruimeng.a100-16-1, 900G/1.31T -> OOM
sh run/cc_v2/gpu16/cc.hybrid_rc20gen80.inbatch.deberta-base.bs4096.gpu16.sh  # pod/sfr-pod-ruimeng.a100-16-0, backup
sh run/cc_v2/gpu16/cc.hybrid_rc20gen80.inbatch+mlp.roberta-base.bs1048.gpu8.sh  # pod/sfr-pod-ruimeng.a100-8-0, backup
sh run/cc_v2/gpu16/cc.hybrid_rc20gen80.inbatch+mlp.roberta-base.bs4096.gpu16.sh  # pod/sfr-pod-ruimeng.a100-16-1
sh run/cc_v2/gpu16/cc.hybrid_rc20gen80.inbatch.roberta-base.bs4096.gpu16.sh  # pod/sfr-pod-ruimeng.a100-16-0
sh run/cc_v2/gpu16/wiki+cc.moco2e14.hybrid_rc20gen80.bs4096.gpu16.sh  # pod/sfr-pod-ruimeng.a100-16-1
sh run/cc_v2/hybrid/cc.moco.T0gen.bs4096.gpu8.sh  # pod/sfr-pod-ruimeng.a100-8-0,backup
sh run/cc_v2/gpu16/cc.moco2e16.hybrid_rc20gen80.roberta.bs8192.gpu16.sh  # pod/sfr-pod-ruimeng.a100-16-2
sh run/cc_v2/gpu16/cc.moco2e14.hybrid_rc20gen80.roberta-large.bs2048.gpu16.sh  # pod/sfr-pod-ruimeng.a100-16-0
sh run/cc_v2/gpu16/cc.moco2e16.hybrid_rc20gen80.bs8192.interleave.gpu16.sh  # pod/sfr-pod-ruimeng.a100-16-0,backup
sh run/cc_v2/gpu16/cc.moco2e16.hybrid_rc20gen80.bs8192.gpu16.sh  # pod/sfr-pod-ruimeng.a100-16-1

sh run/cc_v2/basic/cc.moco.d2q-t2q50.bs2048.gpu8.sh  # pod/sfr-pod-ruimeng.a100-16-1, eval
sh run/cc_v2/basic/cc.moco.topic50.bs4096.gpu8.sh  # pod/sfr-pod-ruimeng.a100-16-0, eval
sh run/cc_v2/gpu16/cc.moco.RC20+T0gen.bs8192.gpu16.sh  # done
sh run/cc_v2/gpu16/cc.moco2e17.hybrid_rc20gen80.bs4096.gpu16.sh  # pod/sfr-pod-ruimeng.a100-8-0
sh run/cc_v2/hybrid/cc.moco.RC20+T0gen.bs2048.gpu8.sh  # pod/sfr-pod-ruimeng.a100-8-0
sh run/cc_v2/basic/cc.moco.topic50.bs2048.gpu8.sh  # done

sh run/cc_v2/gpu16/cc.moco2e14.hybrid_rc20gen80.large.bs2048.lr1e5.gpu16.sh

cd /export/home/project/search/uir_best_cc
sh run/cc_v2/gpu16/cc.moco2e14.hybrid_rc20gen80.large.bs2048.gpu16.sh  # pod/sfr-pod-ruimeng.a100-16-0, eval
# ERROR:torch.distributed.elastic.multiprocessing.api:failed (exitcode: -9) local_rank: 8 (pid: 1125612) of binary. 
#   it's likely to be because that we concatenate train_dataset multiple times (hf_dataloader L220) and it takes too long when batch_size is large. Now disable concatenation and check if newer version of huggingface dataset resolved this issue (NO, it didn't)
sh run/cc_v2/gpu16/cc.moco2e14.hybrid_rc20gen80.bs4096.gpu16.sh  # pod/sfr-pod-ruimeng.a100-16-0, Mem 523G/1.31T
sh run/cc_v2/gpu16/cc.moco2e14.hybrid_rc20gen80.bs8192.gpu16.sh  # pod/sfr-pod-ruimeng.a100-16-0
sh run/cc_v2/gpu16/cc.moco2e14.hybrid_rc20gen80.large.cls.bs2048.gpu16.sh  # done
sh run/cc_v2/gpu16/cc.moco2e14.topic50.large.bs2048.gpu16.sh
sh run/cc_v2/gpu16/cc.moco2e14.topic50.bs8192.gpu16.sh



## CC&wiki DA
cd /export/home/project/search/uir_best_cc
sh run/cc_v1/domain-adapt/cc.T0topic.inbatch.bs1024.gpu8.da-cqa_quora.1e5.step5k.sh  # a100-8-6


sh run/cc_v1/domain-adapt/cc.T0topic.inbatch.bs1024.gpu8.da-sci.5e6.step2k.sh  # a100-8
sh run/cc_v1/domain-adapt/cc.T0topic.inbatch.bs1024.gpu8.da-scifact.5e6.step2k.sh  # a100-8-6
sh run/cc_v1/domain-adapt/cc.T0topic.inbatch.bs1024.gpu8.da-nfcorpus.5e6.step2k.sh  # a100-8-0
sh run/cc_v1/domain-adapt/cc.T0topic.inbatch.bs1024.gpu8.da-scidocs.5e6.step2k.sh  # a100-8 (backup)

sh run/cc_v1/domain-adapt/cc.T0topic.inbatch.bs1024.gpu8.da-touche2020.5e6.step2k.sh  # a100-8
sh run/cc_v1/domain-adapt/cc.T0topic.inbatch.bs1024.gpu8.da-trec_covid.5e6.step2k.sh  # a100-8-0
sh run/cc_v1/domain-adapt/cc.T0topic.inbatch.bs1024.gpu8.da-arguana.5e6.step2k.sh # a100-8-6
sh run/cc_v1/domain-adapt/cc.T0topic.inbatch.bs1024.gpu8.da-fiqa.5e6.step2k.sh  # a100-8 (backup)

sh run/cc_v1/domain-adapt/cc.T0topic.inbatch.bs1024.gpu8.da-scifact.1e5.step5k.sh  # a100-8 (backup)
sh run/cc_v1/domain-adapt/cc.T0topic.inbatch.bs1024.gpu8.da-arguana.1e5.step5k.sh  # done
sh run/cc_v1/domain-adapt/cc.T0topic.inbatch.bs1024.gpu8.da-scidocs.1e5.step5k.sh  # a100-8 (backup)
sh run/cc_v1/domain-adapt/cc.T0topic.inbatch.bs1024.gpu8.da-trec_covid.1e5.step5k.sh  # a100-8
sh run/cc_v1/domain-adapt/cc.T0topic.inbatch.bs1024.gpu8.da-touche2020.1e5.step5k.sh  # a100-8 (backup)
sh run/cc_v1/domain-adapt/cc.T0topic.inbatch.bs1024.gpu8.da-nfcorpus.1e5.step5k.sh  # a100-8 (backup)
sh run/cc_v1/domain-adapt/cc.T0topic.inbatch.bs1024.gpu8.da-fiqa.1e5.step5k.sh  # a100-8
sh run/cc_v1/domain-adapt/cc.T0topic.inbatch.bs1024.gpu8.da-sci.1e5.step5k.sh #  need to remove unused columns like authors
sh run/cc_v1/domain-adapt/cc.T0topic.inbatch.bs1024.gpu8.da-wiki.1e5.step5k.sh  # a100-8-6
sh run/cc_v1/domain-adapt/cc.T0topic.inbatch.bs1024.gpu8.da-quora.1e5.step5k.sh  # a100-8-6
sh run/cc_v1/domain-adapt/cc.T0topic.inbatch.bs1024.gpu8.da-cqa.1e5.step5k.sh  # a100-8-6
sh run/cc_v1/domain-adapt/cc.T0topic.inbatch.bs1024.gpu8.da-msmarco.1e5.step1k.sh
sh run/cc_v1/domain-adapt/cc.T0topic.inbatch.bs1024.gpu8.da-msmarco.1e5.step5k.sh # a100-8-1
sh run/cc_v1/domain-adapt/cc.T0topic.inbatch.bs1024.gpu8.da-msmarco.1e5.step2k.sh # a100-8-1
sh run/cc_v1/domain-adapt/cc.T0topic.moco.bs1024.gpu8.da-msmarco.1e5.step2k.sh  # done
sh run/cc_v1/domain-adapt/cc.T0exsum.inbatch.bs1024.gpu8.da-msmarco.1e5.step2k.sh  # a100-8-6
sh run/cc_v1/domain-adapt/cc.T0exsum.moco.bs1024.gpu8.da-msmarco.1e5.step2k.sh  # a100-8-1

sh run/cc_v1/domain-adapt/cc.ExtQ-plm.inbatch.bs1024.gpu8.da-msmarco.1e5.step2k.warmup200.sh  # a100-8-6
sh run/cc_v1/domain-adapt/cc.ExtQ-plm.inbatch.bs1024.gpu8.da-msmarco.1e5.step1k.warmup100.sh  # a100-8
sh run/cc_v1/domain-adapt/cc.ExtQ-plm.inbatch.bs1024.gpu8.da-msmarco.1e5.step5k.sh  # a100-8-1
sh run/cc_v1/domain-adapt/cc.ExtQ-plm.inbatch.bs1024.gpu8.da-msmarco.5e5.sh  # a100-8


## longer training
sh run/cc_v1/hybrid/cc.moco.RC20+title+T0gen+t2q.bs2048.gpu8.sh
sh run/cc_v1/hybrid/cc.moco.RC20+Qext+title+T0gen+t2q.bs1024.gpu8.sh  # A100-8 (backup)
sh run/cc_v1/hybrid/cc.moco.RC20+Qext+title+T0gen.bs1024.step200k.gpu8.sh  # A100-8
sh run/cc_v1/hybrid/cc.moco.RC20+title+T0gen.bs4096.gpu8.sh  # a100-8-6
    TOKENIZERS_PARALLELISM=true, NUM_WORKER=8, 7.93s/it, #thread=104 (actually bs=2048), killed
    TOKENIZERS_PARALLELISM=false, NUM_WORKER=10, 20-25s/it, #thread=120, killed
    TOKENIZERS_PARALLELISM=true, NUM_WORKER=4, 10~16s/it, #thread=72
    TOKENIZERS_PARALLELISM=true, NUM_WORKER=2, s/it, #thread , killed
sh run/cc_v1/hybrid/cc.moco.RC20+title+T0gen.bs2048.gpu8.sh  # a100-8-5
    TOKENIZERS_PARALLELISM=false, NUM_WORKER=4, 3~4s/it, killed
    TOKENIZERS_PARALLELISM=true, NUM_WORKER=8, 3~6s/it, #thread=75, ETA 300h, killed
sh run/cc_v1/hybrid/cc.moco.RC20+T0gen.bs4096.gpu8.sh  # A100-8-1 
    TOKENIZERS_PARALLELISM=true, NUM_WORKER=6, 18.52s/it, #thread=66
    TOKENIZERS_PARALLELISM=true, NUM_WORKER=6, 10~14s/it, #thread=65, gpus often idle
    TOKENIZERS_PARALLELISM=true, NUM_WORKER=10, 17~25s/it, #thread=90~110
    TOKENIZERS_PARALLELISM=false, NUM_WORKER=4, s/it, #thread, killed
sh run/cc_v1/hybrid/cc.moco.RC20+T0gen.bs2048.gpu8.sh  # A100-8-0
    1.5~2.2s/iter, TOKENIZERS_PARALLELISM=false, #worker=8, #threads=95, utility often not full, but becomes 5+s/iter after 140k steps, 2~4s/it around 155k, 6~8s/it around 160k

cd /export/home/project/search/uir_best_cc
sh run/cc_v1/basic/cc.moco.exsum50.bs2048.gpu8.sh  # A100-8, backup
sh run/cc_v1/basic/cc.moco.absum50.bs2048.gpu8.sh  # A100-8-0
sh run/cc_v1/basic/cc.moco.topic50.bs2048.gpu8.sh  # A100-8
sh run/cc_v1/basic/cc.moco.T0title50.bs2048.gpu8.sh
sh run/cc_v1/basic/cc.moco.topic50.bs4096.gpu8.sh  # a100-8-5
    2.0~2.3s/iter, NUM_WORKER=6, #threads=60


## CC hybrid
cd /export/home/project/search/uir_best_cc
sh run/cc_v1/hybrid/cc.moco.RC50+T0gen.bs1024.gpu8.sh
sh run/cc_v1/hybrid/cc.moco.Qext50+title+T0gen.bs1024.gpu8.sh  # a100-8-1
sh run/cc_v1/hybrid/cc.moco.RC50+Qext+title+T0gen.bs1024.gpu8.sh  # A100-8 (backup)
sh run/cc_v1/hybrid/cc.moco.Qext0+title+T0gen.bs1024.gpu8.sh  # a100-8-0
sh run/cc_v1/hybrid/cc.moco.RC20+Qext+title+T0gen.bs1024.gpu8.sh  # A100-8-1
sh run/cc_v1/hybrid/cc.moco.RC20+T0gen.bs1024.gpu8.sh  # A100-8-0
sh run/cc_v1/hybrid/cc.inbatch.Qext50+title+T0gen.bs1024.gpu8.sh  # A100-8 (backup)
sh run/cc_v1/hybrid/cc.inbatch.Qext0+title+T0gen.bs1024.gpu8.sh  # A100-8
sh run/cc_v1/hybrid/cc.special80.title+T0gen.moco.bs1024.gpu8.sh  # A100-8-2 (backup)
sh run/cc_v1/hybrid/cc.special50.title+T0gen.moco.bs1024.gpu8.sh  # A100-8-3 (backup)


# Fine-tune CC
cd /export/home/project/search/uir_best_cc
sh run/cc_v1/hybrid/cc.moco.RC20+T0gen.bs2048.gpu8.ft+random.sh
sh run/cc_v1/basic/cc.inbatch.absum.bs1024.gpu8.ft+random.sh
sh run/cc_v1/basic/cc.moco.absum50.bs1024.gpu8.ft+random.sh

sh run/cc_v1/hybrid/cc.moco.RC50+Qext+title+T0gen.bs1024.gpu8.ft+random.sh
sh run/cc_v1/hybrid/cc.moco.special80.title+T0gen.bs1024.gpu8.ft+random.sh
sh run/cc_v1/hybrid/cc.moco.special50.title+T0gen.bs1024.gpu8.ft+random.sh

sh run/cc_v1/basic/cc.moco.bs1024.gpu8.ft+random.sh  # a100-8
sh run/cc_v1/basic/cc.inbatch.bs1024.gpu8.ft+random.sh  # a100-8 (backup)
sh run/cc_v1/basic/cc.inbatch.T0title.bs1024.gpu8.ft+random.sh  # done
sh run/cc_v1/basic/cc.moco.T0title50.bs1024.gpu8.ft+random.sh  # a100-8 (backup)
sh run/cc_v1/hybrid/cc.moco.RC50+T0gen.bs1024.gpu8.ft+random.sh  # N/A
sh run/cc_v1/hybrid/cc.moco.RC20+T0gen.bs1024.gpu8.ft+random.sh  # a100-8-5
sh run/cc_v1/hybrid/cc.moco.RC20+Qext+title+T0gen.bs1024.gpu8.ft+random.sh  # a100-8
sh run/cc_v1/basic/cc.moco.topic50.bs2048.gpu8.ft+random.sh  # a100-8-5
sh run/cc_v1/basic/cc.moco.exsum50.bs1024.gpu8.ft+random.sh  # a100-8
sh run/cc_v1/basic/cc.inbatch.exsum.bs1024.gpu8.ft+random.sh  # a100-8 (backup) 

sh run/cc_v1/basic/cc.inbatch.title.bs1024.gpu8.ft+random.sh  # a100-8 (backup)
sh run/cc_v1/basic/cc.moco.title05.bs1024.gpu8.ft+random.sh  # a100-8 (backup)

sh run/cc_v1/qext/cc.ExtQ-plm.inbatch.bs1024.gpu8.ft+random.sh  # a100-8-6
sh run/cc_v1/qext/cc.ExtQ50-plm.moco.bs1024.gpu8.ft+random.sh  # a100-8
sh run/cc_v1/qext/cc.ExtQ-plm.inbatch.bs1024.gpu8.ft+random.5e5.sh  # a100-8-6
sh run/cc_v1/qext/cc.ExtQ-plm.inbatch.bs1024.gpu8.ft+random.2e5.sh  # a100-8-1
sh run/cc_v1/qext/cc.ExtQ-plm.inbatch.bs1024.gpu8.ft+random.5e6.sh  # a100-8-1

sh run/cc_v1/basic/cc.inbatch.topic.bs1024.gpu8.ft+random.sh  #  a100-8
sh run/cc_v1/basic/cc.moco.topic50.bs1024.gpu8.ft+random.sh  # a100-8 (running, q64d192)

sh run/cc_v1/basic/cc.inbatch.d2q-t2q.bs1024.gpu8.ft+random.sh  # a100-8 (backup)
sh run/cc_v1/basic/cc.moco.d2q-t2q50.bs1024.gpu8.ft+random.sh  # a100-8-5


## CC exp
cd /export/home/project/search/uir_best_cc
sh run/cc_v1/basic/cc.moco.exsum50.bs1024.gpu8.sh # A100-8-6, 
TOKENIZERS_PARALLELISM=true, NUM_WORKER=6, 1.96it/s, #thread=89
TOKENIZERS_PARALLELISM=true, NUM_WORKER=2, 1.44it/s 
TOKENIZERS_PARALLELISM=false, NUM_WORKER=0, 2.32s/it
sh run/cc_v1/basic/cc.moco.d2q-t2q50.bs1024.gpu8.sh # A100-8-1, 
TOKENIZERS_PARALLELISM=false, NUM_WORKER=6, 1.93it/s, #thread=65
TOKENIZERS_PARALLELISM=false, NUM_WORKER=2, 1.08it/s
sh run/cc_v1/basic/cc.moco.T0title50.bs1024.gpu8.sh # A100-8 (backup) 
TOKENIZERS_PARALLELISM=true, NUM_WORKER=4, 2.07it/s, #thread=52
sh run/cc_v1/basic/cc.moco.topic50.bs1024.gpu8.sh # A100-8
TOKENIZERS_PARALLELISM=false, NUM_WORKER=4, 2.00it/s, #thread=
sh run/cc_v1/basic/cc.moco.absum50.bs1024.gpu8.sh



sh run/cc_v1/qext/cc.ExtQ50-plm.inbatch.bs1024.gpu8.sh
sh run/cc_v1/qext/cc.ExtQ50-plm.inbatch.bs1024.gpu8.sh  # a100-8-0
sh run/cc_v1/basic/cc.moco.absum50.bs1024.gpu8.sh # A100-8
sh run/cc_v1/basic/cc.inbatch.absum.bs1024.gpu8.sh # done
sh run/cc_v1/basic/cc.moco.topic10.bs1024.gpu8.sh  # a100-8-5
sh run/cc_v1/basic/cc.moco.topic90.bs1024.gpu8.sh  # a100-8-6
sh run/cc_v1/basic/cc.moco.T0title50.bs1024.gpu8.sh  # a100-8
sh run/cc_v1/basic/cc.inbatch.T0title.bs1024.gpu8.sh # a100-8-0
sh run/cc_v1/basic/cc.moco.exsum50.bs1024.gpu8.sh # done
sh run/cc_v1/qext/cc.ExtQ50-plm.moco.bs1024.gpu8.rerun.sh  # a100-8-6
sh run/cc_v1/basic/cc.moco.topic75.bs1024.gpu8.sh  # A100-8 (backup)
sh run/cc_v1/basic/cc.moco.topic25.bs1024.gpu8.sh  # a100-8-6
sh run/cc_v1/qext/cc.ExtQ-plm.moco.bs1024.gpu8.sh  # a100-8-0
sh run/cc_v1/qext/cc.ExtQ-plm.inbatch.bs1024.gpu8.rerun.sh  # a100-8-1

sh run/cc_v1/basic/cc.inbatch.exsum.bs1024.gpu8.sh  # a100-8
sh run/cc_v1/basic/cc.moco.exsum50.bs1024.gpu8.sh  # a100-8-5

sh run/cc_v1/basic/cc.inbatch.title.bs1024.gpu8.sh  # a100-8-6
sh run/cc_v1/basic/cc.inbatch.d2q-t2q.bs1024.gpu8.sh  # a100-8-5

sh run/cc_v1/qext/cc.ExtQ-bm25.inbatch.bs1024.gpu8.sh  # running
sh run/cc_v1/qext/cc.ExtQ-bm25.moco.bs1024.gpu8.sh  # a100-8-1
sh run/cc_v1/qext/cc.ExtQ-plm.inbatch.bs1024.gpu8.sh  # a100-8-6

sh run/cc_v1/basic/cc.inbatch.topic.bs1024.gpu8.sh  # A100-8-1
sh run/cc_v1/basic/cc.inbatch.topic05.bs1024.gpu8.sh  # A100-8-1 (backup)
sh run/cc_v1/basic/cc.moco.2e14.topic.bs1024.gpu8.sh  # A100-8-5

sh run/cc_v1/basic/cc.inbatch.d2q-t2q05.bs1024.gpu8.sh  # A100-8-0
sh run/cc_v1/basic/cc.moco.2e14.d2q-t2q05.bs1024.gpu8.sh  # A100-8-0 (backup)
sh run/cc_v1/cc.moco.2e14.doc2query05-t2q.bs1024.gpu8.sh  # A100-8-1 (backup)

sh run/cc_v1/pile6.moco.2e14.title05.bs2048.gpu8.sh  # A100-8-1, 4workers
sh run/cc_v1/pile10.moco.2e14.title05.bs2048.gpu8.sh  # A100-8-5, 8workers

sh run/cc_v1/cc.moco.2e14.topic05.bs1024.gpu8.sh  # A100-8-3 (backup)
sh run/cc_v1/cc.moco.2e14.title05.bs1024.gpu8.sh  # A100-8-1
sh run/cc_v1/cc.inbatch.title05.bs1024.gpu8.sh  # A100-8-6
sh run/cc_v1/cc.inbatch.doc2query05-t2q.bs1024.gpu8.sh  # A100-8
sh run/cc_v1/cc.inbatch.bs1024.gpu8.sh  # A100-8-1
sh run/cc_v1/cc.moco.2e14.bs1024.gpu8.sh  # A100-8-1 (backup)


# Fine-tune Baseline (SPAR-CLS doesn't work)
cd /export/home/project/search/uir_best_cc
sh run/wikipsg_v1/finetune/contriever.avg.inbatch-random-neg1023+1024.gpu8.sh
sh run/wikipsg_v1/finetune/spar-wiki-query.avg.mm.inbatch-random-neg1023+1024.gpu8.sh  # done
sh run/wikipsg_v1/finetune/contriever.avg.inbatch-random-neg1023+1024.lr5e5.gpu8.sh
sh run/wikipsg_v1/finetune/spider.avg.inbatch-random-neg1023+1024.gpu8.sh  # a100-8-5
sh run/wikipsg_v1/finetune/spar-wiki-context.avg.mm.inbatch-random-neg1023+1024.gpu8.sh  # a100-8-5


# overlap ablation
cd /export/home/project/search/uir_best_cc
sh run/cc_v1/ablate_overlap/cc.ctx128.moco.bs1024.gpu8.sh
sh run/cc_v1/ablate_overlap/cc.ctx256.moco.bs1024.gpu8.sh
sh run/cc_v1/ablate_overlap/cc.ctx512.moco.bs1024.gpu8.sh
sh run/cc_v1/ablate_overlap/cc.ctx1024.moco.bs1024.gpu8.sh
sh run/cc_v1/ablate_overlap/cc.ctxmax.moco.bs1024.gpu8.sh

# length ablation
cd /export/home/project/search/uir_best_cc
sh run/cc_v1/ablate_len/cc.len64.moco.title05.bs1024.gpu8.sh  # A100-8-1 (backup)
sh run/cc_v1/ablate_len/cc.len128.moco.title05.bs1024.gpu8.sh  # A100-8-6
sh run/cc_v1/ablate_len/cc.len384.moco.title05.bs1024.gpu8.sh  # A100-8-6
sh run/cc_v1/ablate_len/cc.len32.moco.title05.bs1024.gpu8.sh  #
sh run/cc_v1/ablate_len/cc.len256.moco.title05.bs1024.gpu8.sh  #

# Finetune wikipsg
cd /export/home/project/search/uir_best_cc
sh run/wikipsg_v1/done/contriever256-special50/wiki.T03b_topic50.moco.bs1024.gpu8.ft+random.sh  # A100-8-6
sh run/wikipsg_v1/finetune/mm.inbatch-random-neg1023+1024.contriever.gpu8.sh  # A100-8-2 (backup)
sh run/wikipsg_v1/done/contriever256-special50/wiki.title50.moco.bs1024.gpu8.ft+random.sh  # A100-8-0 (backup)
sh run/wikipsg_v1/done/contriever256-special50/wiki.title50.moco.bs1024.gpu8.ft.sh  # A100-8-2 (backup)

# wikipsg
cd /export/home/project/search/uir_best_cc
sh run/wikipsg_v1/done/contriever256-special/wiki.extphrase3-50.inbatch.bs1024.gpu8.sh  # A100-8 (backup)
sh run/wikipsg_v1/done/contriever256-special/wiki.extphrase3.moco.bs1024.gpu8.sh  # a100-8-1
sh run/wikipsg_v1/done/contriever256-special/wiki.extphrase3-50.moco.bs1024.gpu8.sh  # a100-8-0
sh run/wikipsg_v1/done/contriever256-special/wiki.extphrase3.inbatch.bs1024.gpu8.sh  # A100-8-2 (backup) 


sh run/wikipsg_v1/contriever256-Qext/wiki.ExtQ-selfdot.inbatch.bs1024.gpu8.sh  # A100-8-6
sh run/wikipsg_v1/contriever256-Qext/wiki.ExtQ-bm25.inbatch.bs1024.gpu8.sh  # A100-8-1 (backup)
sh run/wikipsg_v1/contriever256-Qext/wiki.ExtQ-plm.inbatch.bs1024.gpu8.sh  # A100-8-0 (backup)
sh run/wikipsg_v1/contriever256-Qext/wiki.ExtQ50-selfdot.inbatch.bs1024.gpu8.sh  # A100-8-3 (backup)
sh run/wikipsg_v1/contriever256-Qext/wiki.ExtQ50-plm.inbatch.bs1024.gpu8.sh  # A100-8-5
sh run/wikipsg_v1/contriever256-Qext/wiki.ExtQ50-bm25.inbatch.bs1024.gpu8.sh  # A100-8-2 (backup)

sh run/wikipsg_v1/contriever256-special/paq-fulldoc.moco.bs1024.gpu8.sh  # A100-8-5
sh run/wikipsg_v1/contriever256-special/paq-fulldoc.inbatch.bs1024.gpu8.sh  # A100-8-0 (backup)


sh run/wikipsg_v1/contriever256-special/paq.moco.bs1024.gpu8.sh  #  A100-8-3 (backup)
sh run/wikipsg_v1/contriever256-special/paq.inbatch.bs1024.gpu8.sh  # A100-8-0 (backup)
sh run/wikipsg_v1/contriever256-special/paq50-fulldoc.inbatch.bs1024.gpu8.sh  # A100-8-2 (backup) 
sh run/wikipsg_v1/contriever256-special/paq50-fulldoc.moco.bs1024.gpu8.sh  # A100-8
sh run/wikipsg_v1/contriever256-special/paq50.moco.bs1024.gpu8.sh  # A100-8-3 (backup)
sh run/wikipsg_v1/contriever256-special/paq50.inbatch.bs1024.gpu8.sh  #  A100-8-6
sh run/wikipsg_v1/contriever256-special/wiki.T03b_exsum.inbatch.bs1024.gpu8.sh  # A100-8-1 (backup)
sh run/wikipsg_v1/contriever256-special/wiki.T03b_exsum.moco.bs1024.gpu8.sh  # A100-8-0 (backup)



sh run/wikipsg_v1/contriever256-special/wiki.T03b_absum.moco.bs1024.gpu8.sh  # A100-8-6
sh run/wikipsg_v1/contriever256-special/wiki.T03b_absum.inbatch.bs1024.gpu8.sh  # A100-8-1

sh run/wikipsg_v1/contriever256-special/wiki.T03b_topic.inbatch.bs1024.gpu8.sh  # A100-8-2
sh run/wikipsg_v1/contriever256-special/wiki.T03b_topic.moco.bs1024.gpu8.sh  # A100-8-1 (backup)

sh run/wikipsg_v1/contriever256-special/wiki.T03b_title.inbatch.bs1024.gpu8.sh  # A100-8-3 (backup)
sh run/wikipsg_v1/contriever256-special/wiki.T03b_title.moco.bs1024.gpu8.sh  #  A100-8-0 (backup)
sh run/wikipsg_v1/contriever256-special/wiki.doc2query-t2q.moco.bs1024.gpu8.sh  # A100-8-0
sh run/wikipsg_v1/contriever256-special/wiki.doc2query-t2q.inbatch.bs1024.gpu8.sh  # A100-8

sh run/wikipsg_v1/wiki.title50.inbatch.bs1024.gpu8.sh  # A100-8
sh run/wikipsg_v1/wiki.title50.moco.bs1024.gpu8.sh  # A100-8-0

sh run/wikipsg_v1/wiki.ExtQ-selfdot.moco.bs1024.gpu8.sh  # A100-8-6
sh run/wikipsg_v1/wiki.ExtQ50-bm25.moco.bs1024.gpu8.sh  # A100-8-1 (backup)
sh run/wikipsg_v1/wiki.ExtQ50-selfdot.moco.bs1024.gpu8.sh  # A100-8-0 (backup)
sh run/wikipsg_v1/wiki.ExtQ50-plm.moco.bs1024.gpu8.sh  # A100-8-2 (backup) running
sh run/wikipsg_v1/wiki.ExtQ-plm.moco.bs1024.gpu8.sh  # A100-8-3 (backup) running



sh run/cc_v1/cc.moco.2e14.title05-augdel02.bs2048.gpu8.sh  # A100-8-0
sh run/cc_v1/cc.moco.2e14.topic05.bs2048.gpu8.sh  # A100-8-1
sh run/cc_v1/cc.moco.2e14.doc2query05-t2q.bs2048.gpu8.sh  # A100-8-2
sh run/wikipsg_v1/wiki.title0.moco.bs1024.gpu8.sh  # A100-8-5
sh run/wikipsg_v1/wiki.title0.inbatch.bs1024.gpu8.sh  # A100-8

sh run/cc_v1/cc.moco.2e14.title05.wikipsg256-title50.bs2048.gpu8.sh  # A100-8-4
sh run/wikipsg_v1/wiki.T03b_topic50.moco.contriever50.bs1024.gpu8.sh  # A100-8-3
sh run/wikipsg_v1/wiki.T03b_topic50.moco.bs1024.gpu8.sh  # A100-8-2
sh run/cc_v1/cc.moco.2e14.title05-aug.bs2048.gpu8.sh  # A100-8-1
sh run/cc_v1/cc.moco.2e14.title05.bs2048.gpu8.sh  # A100-8-0

sh run/wikipsg_v1/wiki.doc2query50.moco.contriever50.bs1024.gpu8.sh  # A100-8-5
sh run/wikipsg_v1/wiki.doc2query50.moco.bs1024.gpu8.sh  # A100-8


sh run/wikipsg_v1/wiki.ExtQ-selfdot.moco.bs1024.gpu8.sh  # A100-8
sh run/wikipsg_v1/cc.moco.2e14.title05.bs4096.gpu8.sh  # A100-8-4
sh run/wikipsg_v1/moco.cc.2e14.title05.bs2048.gpu8.rerun.sh  # A100-8-1
sh run/wikipsg_v1/cc.moco.2e17.bs2048.gpu8.sh  # A100-8-2
sh run/wikipsg_v1/cc.moco.2e14.qd224.bs2048.gpu8.sh  # A100-8-5
sh run/wikipsg_v1/cc.inbatch-indep.title05.bs1024.gpu8.sh  # A100-8
sh run/wikipsg_v1/cc.inbatch-2e14.title05.bs1024.gpu8.sh  # A100-8-5, diverged
sh run/wikipsg_v1/cc.inbatch.title05.bs1024.gpu8.sh  # A100-8-4
sh run/wikipsg_v1/cc.moco.2e14.bs2048.gpu8.sh  # A100-8-1
sh run/wikipsg_v1/cc.inbatch.bs1024.gpu8.sh  # A100-8-2
sh run/wikipsg_v1/wiki_T03b_exsum50.inbatch.bs1024.gpu8.sh  # A100-8-3

# MTEB
cd /export/home/project/search/uir_best_cc
nohup bash run/mteb/run_mteb.sh > run/mteb/nohup.mteb.out 2>&1 &

# eval 4-7: 225584
cd /export/share/ruimeng/project/search/simcse
bash run/eval/beireval.4large_datasets.sh
nohup bash run/eval/beireval.2gpu.0-1.sh > nohup.beireval.2gpu.0-1.out 2>&1 &
nohup bash run/eval/beireval.4gpu.4-7.sh > nohup.beireval.4gpu.4-7.out 2>&1 &
nohup bash run/eval/beireval.4gpu.0-3.sh > nohup.beireval.4gpu.0-3.out 2>&1 &
nohup bash run/eval/beireval.2gpu.2-3.sh > nohup.beireval.2gpu.2-3.out 2>&1 &
nohup bash run/eval/beireval.2gpu.4-5.sh > nohup.beireval.2gpu.4-5.out 2>&1 &
nohup bash run/eval/beireval.2gpu.6-7.sh > nohup.beireval.2gpu.6-7.out 2>&1 &


# Fine Tuning
cd /export/home/project/search/uir_best_cc
sh run/finetune/mine_negative.mm.contriever.sh  # A100-8
sh run/finetune/mine_negative.sh


# Query Generation
cd /export/home/project/search/UPR
nohup bash examples/beir/uqg_topic.sh > /export/home/data/search/upr/beir/uqg_topic.arguana.log 2>&1 &
nohup bash examples/beir/uqg_summary_ext.sh > /export/home/data/search/upr/beir/uqg_summary_ext.log 2>&1 &

sh examples/cc/uqg_doc2query_t2q.sh  # A100-8-0

sh examples/wiki/uqg_doc2query_r2t.sh  # done
sh examples/wiki/uqg_doc2query_a2t.sh  # done
sh examples/wiki/uqg_t5xl_insummary.sh  # done
sh examples/wiki/uqg_doc2query.sh  # done
sh examples/wiki/uqg_summary_ext.sh  # done
sh examples/wiki/uqg_summary_abs.sh  # done
sh examples/wiki/uqg_title.sh  # done
sh examples/wiki/uqg_topic.sh  # done

sh examples/cc/uqg_doc2query_a2t.sh  # pod/sfr-pod-ruimeng.a100-16-72cpu-1
sh examples/cc/uqg_doc2query_r2t.sh  # pod/sfr-pod-ruimeng.a100-16-72cpu-0
sh examples/uqg_title.sh  # A100-8-5 (done)
sh examples/uqg_topic.sh  # A100-8-6 (done)
sh examples/uqg_summary_ext.sh  # A100-8-4 (done)
sh examples/uqg_summary_abs.sh  # A100-8-3 (done)

#### extra eval
cd /export/home/project/search/uir_best_cc
sh run/wikipsg_v1/eval/wiki.ExtQ-bm25.moco.bs1024.gpu8.sh  # A100-8-6


sh run/wikipsg_v1/eval/cc.moco.2e17.bs2048.gpu8.sh
sh run/wikipsg_v1/eval/cc.moco.2e14.qd224.bs2048.gpu8.sh  # A100-8-0
sh run/wikipsg_v1/eval/moco.cc.2e14.title05.bs2048.gpu8.rerun.sh  # A100-8-1


sh run/wikipsg_v1/eval/wiki_allphrase5.inbatch.bs1024.gpu8.sh  # A100-8-3
sh run/wikipsg_v1/wiki_T03b_exsum50.moco.bs1024.gpu8.sh  # A100-8-5
sh run/wikipsg_v1/eval/cc.moco.2e14.bs1024.gpu8.sh  # A100-8
sh run/wikipsg_v1/eval/wiki_allphrase5.inbatch.bs1024.gpu8.sh  # A100-8
sh run/wikipsg_v1/eval/wiki_allphrase3.moco.bs1024.gpu8.sh  # A100-8
sh run/wikipsg_v1/eval/wiki_allphrase3.inbatch.bs1024.gpu8.sh  # 
sh run/wikipsg_v1/eval/wiki_T03b_absum50.moco.bs1024.gpu8.sh  # A100-8-4
sh run/wikipsg_v1/eval/wiki_allphrase1.moco.bs1024.gpu8.sh  # A100-8-0, done
sh run/wikipsg_v1/eval/wiki_allphrase1.inbatch.bs1024.gpu8.sh  # done
sh run/wikipsg_v1/eval/wiki_allphrase5.moco.bs1024.gpu8.sh  # A100-8-2
sh run/wikipsg_v1/eval/wiki_T03b_topic50.inbatch.bs1024.gpu8.sh  # done
sh run/wikipsg_v1/cc+wiki.inbatch.title05.bs1024.gpu8.sh  # A100-8 running
sh run/wikipsg_v1/wiki_T03b_exsum.moco.bs1024.gpu8.sh  # A100-8-5
sh run/wikipsg_v1/wiki_T03b_absum50.inbatch.bs1024.gpu8.sh  # A100-8-4
sh run/wikipsg_v1/wiki_T03b_topic50.moco.bs1024.gpu8.sh  # A100-8-3
sh run/wikipsg_v1/wiki_T03b_title50.inbatch.bs1024.gpu8.sh  # A100-8-2
sh run/wikipsg_v1/wiki_T03b_title50.moco.bs1024.gpu8.sh
sh run/wikipsg_v1/wiki_doc2query50.moco.bs1024.gpu8.sh
sh run/wikipsg_v1/wiki_doc2query50.inbatch.bs1024.gpu8.sh  # A100-8-5
sh run/wikipsg_v1/cc+wiki.moco.2e14.title05.bs2048.gpu8.sh  # A100-8-0
sh run/wikipsg_v1/wiki.contriever256.moco.inbatch+2e14.title0.qd128.bs1024.gpu8.sh  # A100-8-2
sh run/wikipsg_v1/wiki.contriever256.moco.inbatch.title0.qd128.bs1024.gpu8.sh  # A100-8-3
sh run/wikipsg_v1/wiki.moco.contriever256.2e17.title0.qd128.bs1024.gpu8.sh  # A100-8-4


sh run/wikipsg_v1/wiki_T03b_exsum.inbatch.bs1024.gpu8.sh  # A100-8-1
sh run/wikipsg_v1/moco.contriever256.wiki.2e17.title0.qd128.bs1024.gpu8.sh  # A100-8-4




sh run/wikipsg_v1/wiki_T03b_absum.moco.bs1024.gpu8.sh  # done
sh run/wikipsg_v1/wiki_doc2query.moco.bs1024.gpu8.sh  # done
sh run/wikipsg_v1/wiki_doc2query.inbatch.bs1024.gpu8.sh  # A100-8-3
sh run/wikipsg_v1/wiki_T03b_absum.inbatch.bs1024.gpu8.sh  # done
sh run/wikipsg_v1/wiki_T03b_absum.moco.bs1024.gpu8.sh  # A100-8-3
sh run/wikipsg_v1/wiki_doc2query.moco.bs1024.gpu8.sh  # A100-8-2
sh run/wikipsg_v1/wiki_T03b_title.moco.bs1024.gpu8.sh  # A100-8-5
sh run/wikipsg_v1/wiki_T03b_title.inbatch.bs1024.gpu8.sh  # A100-8-4


sh run/wikipsg_v1/wiki_T03b_topic.moco.bs1024.gpu8.sh  # torun
sh run/wikipsg_v1/wiki_T03b_topic.inbatch.bs1024.gpu8.sh  # torun
sh run/wikipsg_v1/inbatch.contriever256.wiki.title0.qd128.bs1024.gpu8.sh  # torun
sh run/wikipsg_v1/moco.contriever256.wiki.2e14.title0.qd128.bs1024.gpu8.sh  # torun
sh run/wikipsg_v1/moco.cc.2e14.title05+prompt.bs2048.gpu8.sh  # done
sh run/wikipsg_v1/moco.cc.2e14.title05.bs2048.gpu8.sh  # done


sh run/wikipsg_v1/inbatch.contriever256.wiki.title50.qd128.bs1024.gpu8.sh  # A100-8-2
sh run/wikipsg_v1/moco.contriever256.wiki.2e14.title0.qd128.bs1024.gpu8.sh  # A100-8-0
sh run/wikipsg_v1/paq.inbatch.len256.bs1024.gpu8.sh  # A100-8-1
sh run/wikipsg_v1/inbatch.contriever256.wiki.title100.qd128.bs1024.gpu8.sh  # A100-8
sh run/wikipsg_v1/moco.contriever256.wiki.2e14.title50.qd128.bs1024.gpu8.sh  # A100-8-3
sh run/wikipsg_v1/paq.moco.2e14.contriever256.len256.bs1024.gpu8.sh  # A100-8-2
sh run/wikipsg_v1/moco.contriever256.wiki.2e14.title100.qd128.bs1024.gpu8.sh  # A100-8-1

# To reproduce best runs
## Best_cc finetune
cd /export/home/project/search/uir_best_cc
sh run/finetune/mm.inbatch-random-neg1023+1024.contriever.bs1024.qd192.step20k.drop5.lr1e5.gpu8.sh  # A100-8-1
sh run/finetune/mm.inbatch-random-neg1023+1024.contriever.bs1024.qd192.step20k.lr3e6.gpu8.sh  # A100-8-2
sh run/finetune/mm.inbatch-randoeval/beirm-neg1023+1024.contriever.bs1024.qd192.step20k.lr5e6.gpu8.sh  # A100-8-3
sh run/finetune/mm.inbatch-random-neg1023+1024.contriever.bs1024.qd192.step20k.lr5e8.gpu8.sh  # A100-8-4
sh run/finetune/mm.inbatch-random-neg1023+1024.contriever.bs1024.qd192.step20k.drop1.lr1e5.gpu8.sh  # A100-8-0




sh run/finetune/mm.inbatch-random-neg1023+1024.cc-title50.bs1024.qd184.step20k.gpu8.sh  # A100-8-3
sh run/finetune/mm.inbatch-minedneg_top100_01-neg511+512.contriever.bs512.qd256.step5k.lr1e6.gpu8.sh  # A100-8

sh run/finetune/mm.inbatch-random-neg1023+1024.contriever.bs1024.qd192.step20k.gpu8.sh  # A100-8-3
sh run/finetune/mm.inbatch-random-neg511+512.cc-title50.bs512.qd320.step20k.gpu8.sh  # A100-8-4 
sh run/finetune/mm.inbatch-random-neg511+512.contriever.bs512.qd128.step10k.gpu8.sh  # A100-8-2
sh run/finetune/mm.inbatch-minedneg_top5_01-neg511+512.contriever.bs512.qd256.step10k.lr1e6.gpu8.sh  # A100-8-1
sh run/finetune/mm.inbatch-minedneg_top5_01-neg511+512.contriever.bs512.qd256.step10k.lr5e6.gpu8.sh  # A100-8
sh run/finetune/mm.inbatch-random-neg511+512.contriever.bs512.qd192.step10k.gpu8.sh  # A100-8-0
sh run/finetune/mm.inbatch-minedneg_top5_01-neg511+512.contriever.bs512.qd256.step10k.lr5e5.gpu8.sh  # A100-8-4
sh run/finetune/mm.inbatch-minedneg_top5_01-neg511+512.contriever.bs512.qd256.step10k.gpu8.sh  # A100-8-2
sh run/finetune/mm.inbatch-minedneg_top1_01-neg511+512.contriever.bs512.qd256.step10k.gpu8.sh  # A100-8-1




sh run/finetune/mm.inbatch-random-neg511+512.cc-title50.bs512.qd320.step10k.gpu8.sh  # A100-8
sh run/finetune/mm.inbatch-random-neg511+512.contriever.bs512.qd320.step20k.gpu8.sh  # A100-8-0
sh run/finetune/mm.inbatch-random-neg511+512.bert.bs512.qd320.step20k.gpu8.sh  # A100-8-1
sh run/finetune/mm.inbatch-bm25_01-neg511+512.contriever.bs512.qd320.step20k.gpu8.sh  # A100-8-2
sh run/finetune/mm.inbatch-bm25_05-neg511+512.contriever.bs512.qd320.step20k.gpu8.sh  # A100-8-3
sh run/finetune/mm.inbatch-bm25_1-neg511+512.contriever.bs512.qd320.step20k.gpu8.sh  # A100-8-4


sh run/finetune/mm.inbatch-random-neg511+512.contriever.bs512.qd320.step10k.gpu8.sh  # A100-8
sh run/finetune/mm.inbatch-random-neg511+512.contriever.bs512.qd256.step10k.gpu8.sh  # A100-8-0
sh run/finetune/mm.inbatch-bm25-neg511+512.contriever.bs512.qd256.step10k.gpu8.sh  # A100-8-3
sh run/finetune/mm.inbatch-random-neg511+64.contriever.bs512.qd256.step10k.gpu8.sh  # A100-8
sh run/finetune/mm.inbatch-random.contriever.lr1e6.bs512.qd256.step20k.gpu8.sh  # A100-8-0
sh run/finetune/mm.inbatch-only.contriever.cls.bs512.step10k.gpu8.sh  # A100-8
sh run/finetune/mm.inbatch-bm25-first.contriever.bs1024.step20k.gpu8.sh  # A100-8-4
sh run/finetune/mm.inbatch-random.contriever.bs1024.step20k.gpu8.sh  # A100-8-2
sh run/finetune/mm.inbatch-only.contriever.bs1024.step20k.gpu8.sh  # A100-8-4
sh run/finetune/mm.inbatch-mined_neg.cc-title50-bs2048.bs1024.gpu8.sh  # A100-8-4 (done)
sh run/finetune/mm-random.cc-title50-bs2048.bs1024.gpu8.sh  # A100-8-4 (done)
sh run/finetune/inbatch.finetune.mm.bs1024.step20k.gpu8.sh  # A100-8-2 (done)
sh run/finetune/inbatch.finetune.mm.bs1024.step20k.gpu8.sh  # A100-8-2 (done)
sh run/finetune/moco.finetune.paq.bs1024.gpu8.sh  # A100-8-4 (done)
sh run/finetune/inbatch.finetune.paq.bs1024.gpu8.sh  # A100-8-3 (done)


## Best_cc step100k_v2
cd /export/home/project/search/uir_best_cc
sh run/step100k_v2/moco.contriever256.wiki.2e14.title0.qd128.bs1024.gpu8.sh   # A100-8-0
sh run/step100k_v2/moco.contriever256.wiki.2e16.title0.qd128.bs1024.gpu8.sh   # A100-8-1
sh run/step100k_v2/moco.contriever256.wiki.2e12.title0.qd128.bs1024.gpu8.sh   # A100-8


sh run/step100k_v2/moco.contriever256.wiki.2e16.title0.warmup08-stage16-linear.qd128.bs1024.gpu8.sh   # A100-8-2
sh run/step100k_v2/moco.contriever256.wiki.2e16.title0.warmup08-stage32-linear.qd128.bs1024.gpu8.sh   # A100-8-3
sh run/step100k_v2/moco.contriever256.wiki.2e16.title0.warmup08-stage8-linear.qd128.bs1024.gpu8.sh   # A100-8-1
sh run/step100k_v2/moco.contriever256.wiki-dpr.2e14.title100.qd128.bs512.gpu8.sh   # A100-8-0
sh run/step100k_v2/moco.contriever256.wiki-dpr.2e14.title50.qd128.bs512.gpu8.sh   # A100-8
sh run/step100k_v2/moco.contriever256.wiki-dpr.2e14.title0.qd128.bs512.gpu8.sh   # A100-8-2

sh run/step100k_v2/moco.contriever256.wiki.2e17.title0.warmup08-4stages.qd128.bs512.gpu8.sh   # A100-8-1
sh run/step100k_v2/moco.contriever256.wiki.2e17.title0.warmup08-8stages.qd128.bs512.gpu8.sh   # A100-8
sh run/step100k_v2/moco.contriever256.wiki.2e17.title0.momen99.qd128.bs512.gpu8.sh   # A100-8-0
sh run/step100k_v2/moco.contriever256.wiki.2e10.title0.qd128.bs512.gpu8.sh   # A100-8-3
sh run/step100k_v2/moco.contriever256.wiki.2e17.title0.symmetric.qd128.bs512.gpu8.sh   # A100-8
sh run/step100k_v2/moco.contriever256.wiki.2e12.title0.qd128.bs512.gpu8.sh   # A100-8-2
sh run/step100k_v2/moco.contriever256.wiki.2e17.title0.warmup08.qd128.bs512.gpu8.sh   # A100-8-0

sh run/step100k_v2/moco.contriever64.wiki.2e17.title0.qd128.bs2048.gpu8.sh   # A100-8
sh run/step100k_v2/moco.contriever192.wiki.2e17.title0.qd128.bs2048.gpu8.sh   # A100-8-2
sh run/step100k_v2/moco.contriever128.wiki.2e17.title0.qd128.bs2048.gpu8.sh   # A100-8-1
sh run/step100k_v2/moco.contriever256.wiki.2e17.title0.qd128.bs2048.gpu8.sh   # A100-8-3
sh run/step100k_v2/moco.contriever320.wiki.2e17.title0.qd128.bs2048.gpu8.sh   # A100-8-4

sh run/step100k_v2/moco.wiki.2e17.title100.bs512.gpu8.sh   # A100-8-1 (done)
sh run/step100k_v2/moco.wiki.2e17.title0.bs512.gpu8.sh  # A100-8-2 (done)
sh run/step100k_v2/moco.wiki.2e17.title50.bs512.gpu8.sh  # A100-8-1 (done)

## Best_cc step100k_v1
cd /export/home/project/search/uir_best_cc
sh run/step100k_v1/moco.cc.2e17.title50.norm1.bs512.gpu8.sh   # A100-8-4
sh run/step100k_v1/moco.cc.2e17.title50.bs512.polypower5.lr5e5-end1e8.step200k.gpu8.sh   # A100-8-2
sh run/step100k_v1/moco.cc.2e17.title50.bs512.polypower3.lr5e5-end1e8.step200k.gpu8.sh   # A100-8-1


sh run/step100k_v1/moco.cc.2e17.title50.norm10.bs512.gpu8.sh   # A100-8-1
sh run/step100k_v1/moco.cc.2e17.title50.bs512.polypower3.lr5e5-end5e7.step200k.gpu8.sh   # A100-8-4
sh run/step100k_v1/moco.cc.2e17.title50.bs512.polypower3.lr5e5-end5e7.gpu8.sh   # A100-8-2


sh run/step100k_v1/moco.cc.2e17.title50.bs512.polypower6.lr5e5.gpu8.sh   # A100-8-3
sh run/step100k_v1/moco.cc.2e17.title50.bs512.polypower3.lr5e5-end5e7.step200k.gpu8.sh   # A100-8-1


sh run/step100k_v1/moco.cc.2e17.title0.align05.cancel50k.qd128.bs1024.gpu8.sh   # A100-8-3
sh run/step100k_v1/moco.cc.2e17.title0.align05.qd128.bs1024.gpu8.sh   # A100-8-0
sh run/step100k_v1/moco.cc.2e17.title50.align07.bs512.gpu8.sh  # A100-8-0
sh run/step100k_v1/moco.cc.2e17.title50.align03.bs512.gpu8.sh  # A100-8-0 (done)
sh run/step100k_v1/moco.cc.2e17.title50.align05.bs512.gpu8.sh  # A100-8-0
sh run/remodeled/moco.cc.2e17.title50.q_proj_2layers.bs512.gpu8.sh  # A100-8-4
sh run/remodeled/moco.cc.2e17.title50.qmlp.bs512.gpu8.sh  # A100-8-0
sh run/remodeled/moco.cc.2e17.title50.qmlp_weightnorm.bs512.gpu8.sh  # A100-8-3
sh run/remodeled/moco.cc.2e17.title50.align01.bs512.gpu8.sh  # A100-8-2
sh run/remodeled/moco.cc.2e17.title50.align1.bs512.gpu8.sh  # A100-8-1
sh run/remodeled/moco.cc.2e16.title50.bs512.gpu8.sh  # A100-8-4
sh run/remodeled/moco.cc.2e15.title50.bs512.gpu8.sh  # A100-8-2
sh run/remodeled/moco.cc.2e17.title50.bs512.poly.lr3e5.gpu8.sh  # A100-8-1
sh run/remodeled/moco.cc.2e17.title50.bs512.lr1e5.gpu8.sh  # A100-8-3
sh run/remodeled/moco.cc.inbatch+2e14.title05.seed297.bs512.gpu8.sh  # A100-8-0


sh run/remodeled/moco.cc.2e14.title50.momen0.sameqd50.bs512.gpu8.sh  # A100-8-4 (done)
sh run/remodeled/moco.cc.2e14.title50.momen0.sameqd75.bs512.gpu8.sh  # A100-8-1 (done)
sh run/remodeled/moco.cc.2e17.title50.bs2048.gpu8.sh  # A100-8-2 (done)
sh run/remodeled/moco.cc.2e17.title50.bs1024.gpu8.sh  # A100-8-0 (done)
sh run/remodeled/moco.cc.inbatch.title05.seed297.bs512.gpu8.sh  # A100-8-3 (done)
sh run/remodeled/moco.cc.inbatch+2e17.title05.seed297.bs512.gpu8.sh  # A100-8-1 (done)
sh run/remodeled/moco.cc.2e14.title05.symmetric.seed297.bs512.gpu8.sh  # A100-8-2 (done)
sh run/remodeled/moco.cc.2e17.title50.bs512.gpu8.sh  # A100-8-0 (done)
sh run/remodeled/moco.cc.2e14.title05.indep_encoder_k.seed297.bs512.gpu8.sh  # A100-8-4 (done)
sh run/remodeled/moco.cc.qk2e14.title05.seed297.bs512.gpu8.sh  # A100-8-1 (done)
sh run/remodeled/moco.cc.2e14.title05.interleaved.seed297.bs512.gpu8.sh  # A100-8-2 (done)
sh run/remodeled/moco.cc.2e14.title50.bs512.gpu8.sh  # A100-8-0 (done)
sh run/remodeled/moco.cc.2e17.contriever.bs2048.gpu8.sh  # A100-8 (done)

sh run/remodeled/moco.cc.2e14.prompt+title50.momen90.bs512.gpu8.sh  # A100-8-4 (done)
sh run/remodeled/moco.cc.2e14.prompt+title50.momen50.bs512.gpu8.sh  # A100-8-3 (done)
sh run/remodeled/moco.cc.2e14.prompt+title50.momen10.bs512.gpu8.sh  # A100-8-1 (done)
sh run/remodeled/moco.cc.2e14.prompt+title50.momen0.bs512.gpu8.sh  # A100-8-0 (done)
sh run/remodeled/moco.cc.2e14.large.cls.title05.noprompt.seed297.bs512.gpu8.sh  # A100-8-2 (done)


sh run/remodeled/moco.cc.2e14.title05.noprompt.seed297.bs512.gpu8.sh  # A100-8-0 (done)
sh run/remodeled/moco.cc.2e14.prompt+title50.bs512.gpu8.sh  # A100-8-3 (done)


cd /export/home/project/search/uir_best_cc
sh run/step100k/moco.cc.2e14.title05.bs2048.gpu8.sh  # A100-8-4 (almost done)
sh run/step100k/moco.cc.2e14.prompt+title75.bs512.gpu8.sh  # A100-8-1 (running)
sh run/step100k/moco.cc.2e14.title05.noprompt.seed297.bs512.gpu8.sh  # A100-8-0 (killed)
sh run/step100k/moco.cc.2e14.prompt+title05+del02.bs1024.gpu8.sh  # A100-8-4 (done)
sh run/step100k/moco.cc.2e14.prompt+title05+del02.bs512.gpu8.sh  # A100-8-1 (done)
sh run/step100k/moco.cc.2e14.prompt+title05.qd-tokenize-together.bs512.gpu8.sh  # A100-8-4 (running, seed297)
sh run/step100k/moco.cc.2e14.prompt+title05.noprompt.seed297.bs512.gpu8.sh  # A100-8-0 (running)
sh run/step100k/moco.cc.2e14.prompt+title05.bs2048.gpu8.sh  # A100-8-1, seed119 (running)
sh run/step100k/moco.cc.2e14.prompt+title05.bs4096.gpu8.sh  # A100-8-2, real 4096 (running)
sh run/step100k/moco.cc.2e14.prompt+title05.clip05.bs512.gpu8.sh  # A100-8-3 (done)
sh run/step100k/moco.cc.2e14.prompt+title05.noprompt.seed137.bs512.gpu8.sh  # A100-8-0 (done)
sh run/step100k/moco.cc.2e14.prompt+title05.qd-tokenize-together.bs512.gpu8.sh  # A100-8-6 (done, bad)
sh run/step100k/moco.cc_topic.2e14.prompt+title05.bs512.gpu8.sh  # A100-8-5 (done)
sh run/step100k/moco.cc.2e14.prompt+title05.bs1024.gpu8.sh  # A100-8 (done)
sh run/step100k/moco.cc.2e14.prompt+title05.bs4096.gpu8.sh  # A100-8-0, it's 2048 actually (done)



sh run/step100k/moco.cc.2e14.prompt+title05.noprompt.bs512.gpu8.sh  # A100-8-2 (done)
sh run/step100k/moco.cc.2e14.noprompt.bs2048.gpu8.sh  # A100-8-1 (done)
sh run/step100k/moco.cc.2e14.noprompt.bs512.gpu8.sh  # A100-8 (done)

sh run/step100k/moco.cc.2e14.prompt+title05.bs2048.gpu8.sh  # A100-8-0 (done)


sh run/step100k/moco.cc.2e14.bs512.gpu8.sh  # A100-8 (done)
sh run/step100k/moco.cc.2e14.prompt+title05.bs512.gpu8.sh  # A100-8 (done)
sh run/step100k/moco.interleave.cc.2e14.prompt+title05.bs512.gpu8.sh  # A100-8-5 (done)
sh run/step100k/moco.interleave.cc+wiki.2e14.prompt+title05.bs512.gpu8.sh  # A100-8-1 (done)
sh run/step100k/moco.cc.2e14.prompt.bs2048.gpu8.sh  # A100-8-2 (done)
sh run/step100k/moco.cc.2e14.prompt.bs512.gpu8.sh  # A100-8 (done)
sh run/step100k/moco.cc.2e14.prompt+title05.bs1024.gpu8.sh  # A100-8-6 (done)
sh run/step100k/moco.cc.2e14.prompt+title05.cls.bs512.gpu8.sh  # A100-8-1 (done)

sh run/step100k/moco.cc.2e16.prompt+title05.bs512.gpu8.sh  # A100-8-5 (done)
sh run/step100k/moco.cc_exsum.2e14.prompt+title05.bs512.gpu8.sh  # A100-8-2 (done)
sh run/step100k/moco.cc.2e14.prompt+title1.bs512.gpu8.sh  # A100-8-0 (done)
sh run/step100k/moco.cc_title.2e14.prompt+title05.bs512.gpu8.sh  # A100-8 (done)
sh run/step100k/moco.cc+cc.2e14.prompt+title0.5.v10code+v1title.shuffletwice.gpu8.sh  # title-crop, A100-8 (done)
sh run/step100k/moco.cc+cc.2e14.prompt+title0.5.cosine.gpu8.sh  # A100-8-1 (done)
sh run/step100k/moco.cc.2e14.prompt+topic05.bs512.gpu8.sh  # A100-8-6 (done)
sh run/step100k/moco.cc.2e14.prompt+absum05.bs512.gpu8.sh  # A100-8-5 (done)
sh run/step100k/moco.cc+cc.2e14.prompt+title0.5.v10code+v10title.shuffletwice.gpu8.sh  # title-crop, A100-8-0 (running)
sh run/step100k/moco.cc+cc.2e14.prompt+title0.5.v10code+v1title.shuffletwice.gpu8.sh  # no-title-crop, A100-8-5 (done)
sh run/step100k/moco.cc+cc.2e14.prompt+title0.5.v10code+v10title.shuffletwice.gpu8.sh  # no-title-crop, A100-8-6 (done)
sh run/step100k/moco.cc+cc.2e14.prompt+absum0.5.v10_almostall.shuffleonce.gpu8.sh  # A100-8-3 (bad, killed)
sh run/step100k/moco.cc+cc.2e14.prompt+title0.5.v10_almostall.shuffleonce.gpu8.sh  # A100-8-2 (bad, killed)
sh run/step100k/moco.cc+cc.2e14.contriever.v10_trainer+model.shuffleonce.gpu8.sh  # A100-8-0 (running)
sh run/step100k/moco.cc+cc.2e14.prompt+title0.5.v10_trainer+model.shuffletwice.gpu8.sh  # A100-8-1 (running)
sh run/step100k/moco.cc+cc.2e14.prompt+title0.5.v10_trainer+model.shuffleonce.gpu8.sh  # A100-8 (running)


sh run/moco.cc.2e14.prompt+title0.5.v10_trainer+model.gpu8.sh  # A100-8 (same as best_cc, killed)
sh run/moco.cc+cc.2e14.prompt+title0.5.v10_trainer+model.gpu8.sh  # A100-8-2 (same as best_cc, killed)
sh run/moco.cc.2e14.prompt+title0.5.v10model+dropout_noconfigset.gpu8.sh  # A100-8-2 (same as best_cc, killed)
sh run/moco.cc.2e14.prompt+title0.5.v10model+dropout.gpu8.sh  # A100-8 (same as best_cc, killed)
sh run/moco.cc.2e14.prompt+title0.5.v10model.gpu8.sh  # A100-8-2 (good, killed)
sh run/moco.cc.2e14.prompt+title0.5.v10model_v1.gpu8.sh  # A100-8-4 (collapsed, killed)
sh run/moco.cc.2e14.prompt+title0.5.gpu8.sh  # A100-8 (done)

## Best_cc0 w/ different seed
cd /export/home/project/search/uir_best_cc0
sh run/moco.cc.2e14.title05.noprompt.gpu8.step100k.seed297.sh  # A100-8 (killed, same as best_cc.expect_sameas_cc0.seed297.qd-tokenized-together.cc)
sh run/moco.cc.2e14.prompt+title0.5.gpu8.step100k.seed42.sh  # A100-8 (done, rerun with only q0d0)
sh run/moco.cc+cc.2e14.prompt+title0.5.gpu8.seed42.sh  # A100-8 (good, killed)
sh run/moco.cc.2e14.prompt+title0.5.gpu8.seed997.sh  # A100-8-2 (finished)
sh run/moco.cc.2e14.prompt+title0.5.gpu8.seed337.sh  # A100-8-1 (finished)

sh run/moco.cc.2e14.prompt+title0.5.gpu8.seed42.sh  # A100-8 (dead)
sh run/moco.cc.2e14.prompt+title0.5.bs1024.gpu8.sh  # A100-8-6 (dead)
sh run/moco.wiki.2e14.prompt+title0.5.gpu8.seed42.sh  # A100-8-2 (killed)
sh run/moco.cc.2e14.prompt+title0.5.gpu8.seed337.sh  # A100-8-1 (killed)

## Best_cc00 w/ latest data pipeline

# MoCo v3.2
cd /export/share/ruimeng/project/search/simcse
sh run/moco_v3.2/moco.cc+cc.2e14.prompt+title0.5.bs512.seed42.gpu8.sh  # A100-8-4  (killed)

## Best_subpile6
cd /export/share/ruimeng/project/search/uir_best_subpile6
sh run/moco.wiki+subpile5.concat.2e14.prompt+title0.5.bs512.seed337.gpu8.sh  # A100-8-1 (bad, killed)
sh run/moco.wiki+subpile5.concat.2e14.prompt+title0.5.bs512.gpu8.sh  # A100-8-0 (not good enough, killed)
sh run/moco.wiki+subpile5.2e14.prompt+title0.5.bs512.seed337.gpu8.sh  # A100-8-1 (bad, killed)
sh run/moco.cc.2e14.prompt+title0.5.bs512.seed997.gpu8.sh  # A100-8-1 (canceled)
sh run/moco.wiki+subpile5.2e14.prompt+title0.5.bs512.gpu8.sh  # A100-8-0 (killed)
sh run/moco.cc.2e14.prompt+title0.5.bs512.seed337.gpu8.sh  # A100-8-0 (not good, killed)
sh run/moco.cc.2e14.prompt+title0.5.bs512.gpu8.sh  # A100-8-2 (killed,bf16)
sh run/moco.cc.2e14.prompt+title0.5.bs1024.gpu8.sh  # A100-8-1 (killed)






# MoCo v3
cd /export/share/ruimeng/project/search/simcse
sh run/moco_v3.2/moco.cc.2e14.prompt+title0.5.bs512.seed337.gpu8.sh  # A100-8-2  (killed)
sh run/moco_v3.2/moco.cc.2e14.prompt+title0.5.bs512.seed42.gpu8.sh  # A100-8-1  (killed)


sh run/moco_v3.1/moco.cc.2e14.prompt+title0.5.gpu8.bs512.sh  # A100-8
sh run/moco_v3.1/moco.cc.2e17.contriever+title05.step200k.bs2048.gpu8.sh  # A100-8
sh run/moco_v3.1/moco.cc.2e17.contriever.step200k.bs2048.gpu8.sh  # A100-8-1
sh run/moco_v3.1/moco.cc.2e17.contriever.step200k.bs4096.gpu8.sh  # A100-8-0
sh run/moco_v3.1/moco.cc.2e14.contriever+title05.step200k.bs2048.gpu8.sh  # A100-8-4
sh run/moco_v3.1/moco.cc.2e14.contriever.step200k.bs2048.gpu8.sh  # A100-8-6
sh run/moco_v3.1/moco.cc.2e14.prompt+title0.5.gpu8.bs512.sh  # A100-8 (title-retained), A100-8-1 (no-title-retained)
sh run/moco_v3.1/moco.cc.bert_large.2e14.prompt+title0.5.gpu8.bs512.sh  # A100-8-4
sh run/moco_v3.1/moco.cc.2e16.prompt+title0.5.gpu8.bs1024.sh  # A100-8-3
sh run/moco_v3.1/moco.wiki+subpile5.2e14.prompt+title0.5.gpu8.bs512.sh  # A100-8-0 (v10), A100-8-2 (v9)
sh run/moco_v3.1/moco.cc.2e14.prompt+title0.25.gpu8.bs512.sh
sh run/moco_v3.1/moco.wiki+subpile10.2e14.prompt+title0.5.gpu8.bs512.sh
sh run/moco_v3.1/moco.cc.2e14.prompt+QinD01.gpu8.bs512.sh
sh run/moco_v3.1/moco.cc.2e14.prompt+title05.gradclip2.gpu8.bs512.sh

sh run/moco_v3.1/moco.cc.2e14.prompt+title1.gpu8.bs512.sh
sh run/moco_v3.1/moco.cc.2e14.prompt+title0.75.gpu8.bs512.sh
sh run/moco_v3.1/moco.wiki+subpile5.2e14.prompt+title0.5.gpu8.bs512.sh

sh run/moco_v3.1/moco.cc.2e14.prompt+title0.5.gpu8.bs2048.sh
sh run/moco_v3.1/moco.cc.2e14.prompt+title0.5.gpu8.bs4096.sh
sh run/moco_v3.1/moco.cc.2e14.prompt+title0.5.gpu8.bs1024.sh
sh run/moco_v3.1/moco.cc.2e14.prompt+title0.5.gpu8.sh
sh run/moco_v3.1/moco.cc.2e14.prompt+title0.5.amend_v4.gpu8.sh
sh run/moco_v3.1/moco.cc.2e14.prompt+title0.5.worker0.gpu8.sh
sh run/moco_v3.1/moco.wiki+subpile5.2e14.prompt+title0.5.gpu8.sh
sh run/moco_v3.1/moco.wiki+subpile5-dup.2e14.prompt+title0.5.gpu8.sh

# MoCo v2
sh run/moco_v2/moco.cc.prompt+title0.5.bs1024.gpu16.sh
sh run/moco_v2/moco.cc.2e14.prompt+title0.5.gpu8.bs1024-v3.sh
sh run/moco_v2/moco.cc.2e14.prompt+title0.5.gpu8-v3.sh
sh run/moco_v2/moco.cc.2e14.prompt+title0.5.gpu8.sh
sh run/moco_v2/moco.wiki+cc.2e14.prompt.gpu8.sh
sh run/moco_v2/moco.wiki+subpile10.2e14.prompt+title0.5.gpu8.sh
sh run/moco_v2/moco.cc.2e14.prompt.gpu8.sh
sh run/moco_v2/moco.wiki+subpile10.2e14.prompt.gpu8.sh
sh run/moco_v2/moco.wiki+cc.2e14.prompt+title0.5.gpu8.sh
sh run/moco_v2/moco.owt+wiki.2e14.prompt+title0.5.gpu8.sh
sh run/moco_v2/moco.pile.2e14.prompt+title0.5.gpu8.sh
sh run/moco_v2/moco.pile.2e14.prompt+title1.gpu8.sh
sh run/moco_v2/moco.pile.2e14.prompt+title0.gpu8.sh
sh run/moco_v2/moco.pile.2e14.prompt+title0.5.gpu8.test.sh

sh run/moco_v2/moco.pile+wiki.2e14.prompt+title1.gpu8.sh
sh run/moco_v2/moco.wiki.2e14.updatefreq.gpu8.sh
sh run/moco_v2/moco.pile+wiki.2e14.prompt+title0.5.gpu8.sh
sh run/moco_v2/moco.pile+wiki.2e17.gpu8.sh
sh run/moco_v2/moco.wiki.2e14.q128d512.gpu8.sh
sh run/moco_v2/moco.wiki+beir.2e14.gpu8.sh
sh run/moco_v2/moco.wiki.2e14.prompt.gpu8.sh
sh run/moco_v2/moco.pile+wiki.2e14.gpu8.sh

sh run/moco_v2/moco.beir.2e14.gpu8.sh
sh run/moco_v2/moco.wikipedia.2e14.align1.gpu8.sh
sh run/moco_v2/moco.wikipedia.2e14.unif1.gpu8.sh


sh run/moco_v2/moco.c4+wiki.moment099.2e18.gpu16.sh

# MoCo v1
cd /export/share/ruimeng/project/search/simcse
sh run/moco_v1/moco.wikipedia.2e14.warmupqueue.gpu8.sh
sh run/moco_v1/moco.wikipedia.2e14.cls.gpu8.sh
sh run/moco_v1/moco.wikipedia.2e14.lr3e5.gpu8.sh

sh run/moco_v1/moco.moment095.gpu8.sh
sh run/moco_v1/moco.moment0999.gpu8.sh

sh run/moco_v1/moco.c4+wiki.moment0.2e17.bs512.gpu8.sh
sh run/moco_v1/moco.c4+wiki.moment0.2e18.gpu8.sh
sh run/moco_v1/moco.moment025.gpu8.sh
sh run/moco_v1/moco.c4+wiki.moment0.gpu8.sh
sh run/moco_v1/moco.wikipedia.2e8.gpu8.sh
sh run/moco_v1/moco.moment0.gpu8.sh
sh run/moco_v1/moco.pile.gpu8.sh
sh run/moco_v1/moco.dot_normQD.gpu8.sh

sh run/moco_v1/moco.bert-large.gpu8.sh
sh run/moco_v1/moco.bert-large.lr_polynomial_power2.gpu8.sh


sh run/moco_v1/moco.wikipedia.2e14.baseline.gpu8.sh
sh run/moco_v1/moco.step100k.gpu8.cosine.sh
sh run/moco_v1/moco.step100k.gpu8.dual.sh

sh run/moco_v1/moco.step100k.gpu8.moment1.sh
sh run/moco_v1/moco.lr_polynomial_power2.gpu8.sh
sh run/moco_v1/moco.lr_decayed_cosine.gpu8.sh



sh run/moco_v1/moco.step100k.gpu8-1.sh
sh run/moco_v1/moco.constant_with_warmup.gpu8.sh
sh run/moco_v1/moco.gpu8.sh
sh run/moco_v1/moco.gpu4.sh
sh run/moco_v1/moco.gpu2.sh
sh run/moco_v1/moco.gpu1.sh
sh run/moco_v1/moco.step100k.gpu8.sh
sh run/moco_v1/moco.step100k.gpu8-2.sh


cd /export/share/ruimeng/project/search/simcse
sh run/exp_v1/cl.doc.step100k.gpu8.sh
sh run/exp_v1/cl.doc.step100k.gpu8-2.sh
sh run/exp_v1/cl.doc.step100k.gpu8-1.sh
sh run/exp_v1/cl.psg.step100k.gpu8.sh
sh run/exp_v1/cl.psg.step100k.gpu16.sh



sh run/contriever_v2/cl.modelv1.doc.max128.step100k.gpu8.sh
sh run/contriever_v2/cl.psg.step100k.gpu8-2.sh
sh run/contriever_v2/cl.psg.step100k.gpu8.sh
sh run/contriever_v2/cl.psg.step100k.gpu8-1.sh

source run/contriever_v2/cl.doc.step100k.gpu8.sh
source run/contriever_v2/cl.doc.step100k.gpu8-1.sh
source run/contriever_v2/cl.doc.step100k.gpu8-2.sh
source run/contriever_v2/cl.doc.step100k.gpu16.sh


source run/contriever/cl.passage.step100k.gpu8.sh
source run/contriever/cl.passage.step100k.gpu8-1.sh
source run/contriever/cl.passage.step100k.gpu8-2.sh
source run/contriever/cl.passage.step100k.gpu16.sh


source run/contriever/CL_pretrain.doc.step100k.gpu8.sh
source run/contriever/CL_pretrain.doc.step100k.gpu0-7.sh
source run/contriever/CL_pretrain.doc.step100k.gpu8-15.sh
source run/contriever/CL_pretrain.passage.step100k.gpu0-7.sh
source run/contriever/CL_pretrain.passage.step100k.gpu8-15.sh
