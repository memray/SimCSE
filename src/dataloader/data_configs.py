import logging

data_pipelines = {
    'wikipsg128-special50%': {
        'max_context_len': 128,
        'min_dq_len': 4,
        'q_del_ratio': 0.0,
        'd_del_ratio': 0.1,
        'query_column_name': 'special_query',
        'special_query_ratio': 0.5,
        'aug_special_query': False,
    },
    'wikipsg256-special50%': {
        'max_context_len': 256,
        'min_dq_len': 4,
        'q_del_ratio': 0.0,
        'd_del_ratio': 0.1,
        'query_column_name': 'special_query',
        'special_query_ratio': 0.5,
        'aug_special_query': False,
    },
    'wikipsg128-title50%': {
        'max_context_len': 128,
        'min_dq_len': 4,
        'q_del_ratio': 0.0,
        'd_del_ratio': 0.1,
        'query_column_name': 'title',
        'special_query_ratio': 0.5,
        'aug_special_query': False,
    },
    'wikipsg256-title50%': {
        'max_context_len': 256,
        'min_dq_len': 4,
        'q_del_ratio': 0.0,
        'd_del_ratio': 0.1,
        'query_column_name': 'title',
        'special_query_ratio': 0.5,
        'aug_special_query': False,
    },
    'wikipsg256-special': {
        'max_context_len': 256,
        'min_dq_len': 4,
        'q_del_ratio': 0.0,
        'd_del_ratio': 0.1,
        'special_query_ratio': 1.0,
        'query_column_name': 'special_query',
    },
    'contriever256-special': {
        'max_context_len': 256,
        'min_dq_len': 4,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'd_del_ratio': 0.1,
        'special_query_ratio': 1.0,
        'query_column_name': 'special_query',
        'aug_special_query': False,
    },
    'contriever256-special50%': {
        'max_context_len': 256,
        'min_dq_len': 4,
        'min_q_len': 0.05,
        'max_q_len': 0.5,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'q_del_ratio': 0.1,
        'd_del_ratio': 0.1,
        'query_column_name': 'special_query',
        'special_query_ratio': 0.5,
        'aug_special_query': False,
    },
    'contriever256-Qtitle50%': {
        'max_context_len': 256,
        'min_dq_len': 4,
        'min_q_len': 0.05,
        'max_q_len': 0.5,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'q_del_ratio': 0.1,
        'd_del_ratio': 0.1,
        'special_query_ratio': 0.5,
        'query_column_name': 'title',
        'aug_special_query': False,
    },
    'contriever256-Qtitle50%-aug': {
        'max_context_len': 256,
        'min_dq_len': 4,
        'min_q_len': 0.05,
        'max_q_len': 0.5,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'q_del_ratio': 0.1,
        'd_del_ratio': 0.1,
        'special_query_ratio': 0.5,
        'query_column_name': 'title',
        'aug_special_query': True,
    },
    'contriever256-Qtitle50%-aug-del20%': {
        'max_context_len': 256,
        'min_dq_len': 4,
        'min_q_len': 0.05,
        'max_q_len': 0.5,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'q_del_ratio': 0.2,
        'd_del_ratio': 0.2,
        'special_query_ratio': 0.5,
        'query_column_name': 'title',
        'aug_special_query': True,
    },
    'contriever256': {
        'max_context_len': 256,
        'min_dq_len': 4,
        'min_q_len': 0.05,
        'max_q_len': 0.5,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'q_del_ratio': 0.1,
        'd_del_ratio': 0.1,
    },
    'wikipsg256-title': {
        'max_context_len': 256,
        'min_dq_len': 4,
        'q_del_ratio': 0.0,
        'd_del_ratio': 0.1,
        'special_query_ratio': 1.0,
        'query_column_name': 'title',
    },
    'contriever256-Qtitle100%': {
        'max_context_len': 256,
        'min_dq_len': 4,
        'min_q_len': 0.05,
        'max_q_len': 0.5,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'q_del_ratio': 0.1,
        'd_del_ratio': 0.1,
        'special_query_ratio': 1.0,
        'query_column_name': 'title',
    },
    'wikipsg256-allphrase1': {
        'max_context_len': 256,
        'min_dq_len': 4,
        'q_del_ratio': 0.0,
        'd_del_ratio': 0.1,
        'special_query_ratio': 1.0,
        'query_column_name': 'all_phrases',
        'max_phrase_num': 1,
    },
    'wikipsg256-allphrase3': {
        'max_context_len': 256,
        'min_dq_len': 4,
        'q_del_ratio': 0.0,
        'd_del_ratio': 0.1,
        'special_query_ratio': 1.0,
        'query_column_name': 'all_phrases',
        'max_phrase_num': 3,
    },
    'wikipsg256-allphrase5': {
        'max_context_len': 256,
        'min_dq_len': 4,
        'q_del_ratio': 0.0,
        'd_del_ratio': 0.1,
        'special_query_ratio': 1.0,
        'query_column_name': 'all_phrases',
        'max_phrase_num': 5,
    },
    'contriever256-allphrase3-50%': {
        'max_context_len': 256,
        'min_dq_len': 4,
        'min_q_len': 0.05,
        'max_q_len': 0.5,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'q_del_ratio': 0.1,
        'd_del_ratio': 0.1,
        'special_query_ratio': 0.5,
        'aug_special_query': False,
        'query_column_name': 'all_phrases',
        'max_phrase_num': 3,
    },
    'contriever256-allphrase5-50%': {
        'max_context_len': 256,
        'min_dq_len': 4,
        'min_q_len': 0.05,
        'max_q_len': 0.5,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'q_del_ratio': 0.1,
        'd_del_ratio': 0.1,
        'special_query_ratio': 0.5,
        'aug_special_query': False,
        'query_column_name': 'all_phrases',
        'max_phrase_num': 5,
    },
    'asis': {
        'max_context_len': 0,
        'min_dq_len': 1,
        'min_q_len': 1.0,
        'max_q_len': 1.0,
        'min_d_len': 1.0,
        'max_d_len': 1.0,
        'q_del_ratio': 0.0,
        'd_del_ratio': 0.0,
        'special_query_ratio': 0.0,
    },
    'asis-prompt': {
        'max_context_len': 0,
        'min_dq_len': 1,
        'min_q_len': 1.0,
        'max_q_len': 1.0,
        'min_d_len': 1.0,
        'max_d_len': 1.0,
        'q_del_ratio': 0.0,
        'd_del_ratio': 0.0,
        'dq_prompt_ratio': 1.0,
        'special_query_ratio': 0.0,
    },
    'contriever-64': {
        'max_context_len': 64,
        'min_dq_len': 1,
        'min_q_len': 0.05,
        'max_q_len': 0.5,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'q_del_ratio': 0.1,
        'd_del_ratio': 0.1,
    },
    'contriever-128': {
        'max_context_len': 128,
        'min_dq_len': 1,
        'min_q_len': 0.05,
        'max_q_len': 0.5,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'q_del_ratio': 0.1,
        'd_del_ratio': 0.1,
    },
    'contriever-192': {
        'max_context_len': 192,
        'min_dq_len': 1,
        'min_q_len': 0.05,
        'max_q_len': 0.5,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'q_del_ratio': 0.1,
        'd_del_ratio': 0.1,
    },
    'contriever-320': {
        'max_context_len': 320,
        'min_dq_len': 1,
        'min_q_len': 0.05,
        'max_q_len': 0.5,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'q_del_ratio': 0.1,
        'd_del_ratio': 0.1,
    },
    'contriever-Q128D512-prompt': {
        'max_context_len': 512,
        'min_dq_len': 8,
        'min_q_len': 0.05,
        'max_q_len': 0.5,
        'min_d_len': 0.5,
        'max_d_len': 1.0,
        'q_del_ratio': 0.1,
        'd_del_ratio': 0.1,
        'dq_prompt_ratio': 1.0,
        'special_query_ratio': 0.0,
    },
    'contriever256-prompt': {
        'max_context_len': 256,
        'min_dq_len': 8,
        'min_q_len': 0.05,
        'max_q_len': 0.5,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'q_del_ratio': 0.1,
        'd_del_ratio': 0.1,
        'dq_prompt_ratio': 1.0,
        'special_query_ratio': 0.0,
    },
    'contriever256-prompt50%': {
        'max_context_len': 256,
        'min_dq_len': 8,
        'min_q_len': 0.05,
        'max_q_len': 0.5,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'q_del_ratio': 0.1,
        'd_del_ratio': 0.1,
        'dq_prompt_ratio': 0.5,
        'special_query_ratio': 0.0,
    },
    'contriever256-prompt50%-Qtitle50%': {
        'max_context_len': 256,
        'min_dq_len': 8,
        'min_q_len': 0.05,
        'max_q_len': 0.5,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'q_del_ratio': 0.1,
        'd_del_ratio': 0.1,
        'dq_prompt_ratio': 0.5,
        'special_query_ratio': 0.5,
        'query_column_name': 'title',
    },
    'contriever256-prompt-Qtitle25%': {
        'max_context_len': 256,
        'min_dq_len': 8,
        'min_q_len': 0.05,
        'max_q_len': 0.5,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'q_del_ratio': 0.1,
        'd_del_ratio': 0.1,
        'dq_prompt_ratio': 1.0,
        'special_query_ratio': 0.25,
        'query_column_name': 'title',
    },
    'contriever256-prompt-Qtitle50%': {
        'max_context_len': 256,
        'min_dq_len': 8,
        'min_q_len': 0.05,
        'max_q_len': 0.5,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'q_del_ratio': 0.1,
        'd_del_ratio': 0.1,
        'dq_prompt_ratio': 1.0,
        'special_query_ratio': 0.5,
        'query_column_name': 'title',
    },
    'contriever256-prompt-Qtitle75%': {
        'max_context_len': 256,
        'min_dq_len': 8,
        'min_q_len': 0.05,
        'max_q_len': 0.5,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'q_del_ratio': 0.1,
        'd_del_ratio': 0.1,
        'dq_prompt_ratio': 1.0,
        'special_query_ratio': 0.75,
        'query_column_name': 'title',
    },
    'contriever256-prompt-Qtitle100%': {
        'max_context_len': 256,
        'min_dq_len': 8,
        'min_q_len': 0.05,
        'max_q_len': 0.5,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'q_del_ratio': 0.1,
        'd_del_ratio': 0.1,
        'dq_prompt_ratio': 1.0,
        'special_query_ratio': 1.0,
        'query_column_name': 'title',
    },
    'contriever-512-prompt-Qtitle100%': {
        'max_context_len': 512,
        'min_dq_len': 8,
        'min_q_len': 0.05,
        'max_q_len': 0.5,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'q_del_ratio': 0.1,
        'd_del_ratio': 0.1,
        'dq_prompt_ratio': 1.0,
        'special_query_ratio': 1.0,
        'query_column_name': 'title',
    },
    'contriever-512-prompt-Qtitle50%': {
        'max_context_len': 512,
        'min_dq_len': 8,
        'min_q_len': 0.05,
        'max_q_len': 0.5,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'q_del_ratio': 0.1,
        'd_del_ratio': 0.1,
        'dq_prompt_ratio': 1.0,
        'special_query_ratio': 0.5,
        'query_column_name': 'title',
    },
    'contriever256-prompt-QinD10%': {
        'max_context_len': 256,
        'min_dq_len': 8,
        'min_q_len': 0.05,
        'max_q_len': 0.5,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'q_del_ratio': 0.1,
        'd_del_ratio': 0.1,
        'query_in_doc': 0.1,
        'dq_prompt_ratio': 1.0,
        'special_query_ratio': 0.0,
    },
    'contriever256-prompt-QinD50%': {
        'max_context_len': 256,
        'min_dq_len': 8,
        'min_q_len': 0.05,
        'max_q_len': 0.5,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'q_del_ratio': 0.1,
        'd_del_ratio': 0.1,
        'query_in_doc': 0.5,
        'dq_prompt_ratio': 1.0,
        'special_query_ratio': 0.5,
    },
    'ctx256-q30%d50%-prompt-Qtitle50%': {
        'max_context_len': 256,
        'min_dq_len': 8,
        'min_q_len': 0.05,
        'max_q_len': 0.3,
        'min_d_len': 0.05,
        'max_d_len': 0.5,
        'q_del_ratio': 0.1,
        'd_del_ratio': 0.1,
        'dq_prompt_ratio': 1.0,
        'special_query_ratio': 0.5,
        'query_column_name': 'title',
    },
}


def load_dataprocess_config(training_args, hftraining_args):
    logger = logging.getLogger(__name__)
    # prepare for data loader
    if training_args.data_pipeline_name:
        data_prep_config = data_pipelines[training_args.data_pipeline_name]
        if hftraining_args.local_rank == 0 or hftraining_args.local_rank == -1:
            logger.info('Using pre-defined data pipeline: ' + str(training_args.data_pipeline_name))
    else:
        data_prep_config = {
            'max_context_len': training_args.max_context_len,
            'min_dq_len': training_args.min_dq_len,
            'min_q_len': training_args.min_q_len,
            'max_q_len': training_args.q_len,
            'min_d_len': training_args.min_d_len,
            'max_d_len': training_args.max_d_len,
            'word_del_ratio': training_args.word_del_ratio,
            'dq_prompt_ratio': training_args.dq_prompt_ratio,
            'special_query_ratio': training_args.special_query_ratio,
        }
    if hftraining_args.local_rank == 0 or hftraining_args.local_rank == -1:
        logger.info('Data loading parameters:')
        for k, v in data_prep_config.items():
            setattr(training_args, k, v)
            logger.info(f'\t\t{k} = {v}')

    return data_prep_config

