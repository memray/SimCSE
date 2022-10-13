fuser -v /dev/nvidia* | awk '{ print $0 }' | xargs -n1 kill -9

# eval 4-7: 225584
cd /export/share/ruimeng/project/search/simcse
bash run/eval/beireval.4large_datasets.sh
nohup bash run/eval/beireval.2gpu.0-1.sh > nohup.beireval.2gpu.0-1.out 2>&1 &
nohup bash run/eval/beireval.4gpu.4-7.sh > nohup.beireval.4gpu.4-7.out 2>&1 &
nohup bash run/eval/beireval.4gpu.0-3.sh > nohup.beireval.4gpu.0-3.out 2>&1 &
nohup bash run/eval/beireval.2gpu.2-3.sh > nohup.beireval.2gpu.2-3.out 2>&1 &
nohup bash run/eval/beireval.2gpu.4-5.sh > nohup.beireval.2gpu.4-5.out 2>&1 &
nohup bash run/eval/beireval.2gpu.6-7.sh > nohup.beireval.2gpu.6-7.out 2>&1 &


# Query Generation
cd /export/share/ruimeng/project/search/uir_best_cc
sh run/finetune/mine_negative.mm.contriever.sh  # A100-8
sh run/finetune/mine_negative.sh


# Query Generation
cd /export/share/ruimeng/project/search/UPR
sh examples/wiki/uqg_title.sh  # A100-8

sh examples/uqg_title.sh  # A100-8-5 (done)
sh examples/uqg_topic.sh  # A100-8-6 (done)
sh examples/uqg_summary_ext.sh  # A100-8-4 (done)
sh examples/uqg_summary_abs.sh  # A100-8-3 (done)


## New wiki exp
cd /export/share/ruimeng/project/search/uir_best_cc
sh run/wikipsg_v1/moco.contriever256.wiki.2e14.title50.qd128.bs1024.gpu8.sh  # A100-8-3
sh run/wikipsg_v1/inbatch.contriever256.wiki.title100.qd128.bs1024.gpu8.sh  # A100-8-1
sh run/wikipsg_v1/inbatch.contriever256.wiki.title0.qd128.bs1024.gpu8.sh  # A100-8-0
sh run/wikipsg_v1/moco.contriever256.wiki.2e14.title0.qd128.bs1024.gpu8.sh  # A100-8-2
sh run/wikipsg_v1/moco.contriever256.wiki.2e14.title100.qd128.bs1024.gpu8.sh  # A100-8

# To reproduce best runs
## Best_cc finetune
cd /export/share/ruimeng/project/search/uir_best_cc
sh run/finetune/mm.inbatch-random-neg1023+1024.contriever.bs1024.qd192.step20k.drop5.lr1e5.gpu8.sh  # A100-8-1
sh run/finetune/mm.inbatch-random-neg1023+1024.contriever.bs1024.qd192.step20k.lr3e6.gpu8.sh  # A100-8-2
sh run/finetune/mm.inbatch-random-neg1023+1024.contriever.bs1024.qd192.step20k.lr5e6.gpu8.sh  # A100-8-3
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
cd /export/share/ruimeng/project/search/uir_best_cc
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
cd /export/share/ruimeng/project/search/uir_best_cc
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


cd /export/share/ruimeng/project/search/uir_best_cc
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
cd /export/share/ruimeng/project/search/uir_best_cc0
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
