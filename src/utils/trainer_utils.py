# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from transformers.file_utils import ExplicitEnum

class SchedulerType(ExplicitEnum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    COSINE_WITH_DECAYED_RESTARTS = "cosine_with_decayed_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
