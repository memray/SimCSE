fuser -v /dev/nvidia* | awk '{ print $0 }' | xargs -n1 kill -9

# eval 4-7: 225584
cd /export/share/ruimeng/project/search/simcse
nohup bash run/eval/beireval.2gpu.0-1.sh > nohup.beireval.2gpu.0-1.out 2>&1 &
nohup bash run/eval/beireval.4gpu.4-7.sh > nohup.beireval.4gpu.4-7.out 2>&1 &
nohup bash run/eval/beireval.4gpu.0-3.sh > nohup.beireval.4gpu.0-3.out 2>&1 &
nohup bash run/eval/beireval.2gpu.2-3.sh > nohup.beireval.2gpu.2-3.out 2>&1 &
nohup bash run/eval/beireval.2gpu.4-5.sh > nohup.beireval.2gpu.4-5.out 2>&1 &
nohup bash run/eval/beireval.2gpu.6-7.sh > nohup.beireval.2gpu.6-7.out 2>&1 &

# MoCo v3
cd /export/share/ruimeng/project/search/simcse
sh run/moco_v3.1/moco.cc.2e14.prompt+title0.5.worker0.gpu8.sh
sh run/moco_v3.1/moco.cc.2e14.prompt+title0.5.gpu8.sh
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
