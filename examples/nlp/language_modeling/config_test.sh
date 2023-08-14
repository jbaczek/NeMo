set -x

declare -A pairs=(
    [megatron_gpt_pretraining.py]=gpt
    [megatron_bert_pretraining.py]=bert
    [megatron_t5_pretraining.py]=t5
    [megatron_retro_pretraining.py]=retro
    [megatron_bart_pretraining.py]=bart
)

#for prefix in "" alt_
for prefix in alt_
do
mkdir -p config_tests/${prefix}conf
for SCRIPT in  "${!pairs[@]}"
do
    FNAME=megatron_${pairs[$SCRIPT]}_config
    if [ ${prefix} == alt_ ] 
    then
        CONF_ROOT='--config-name=pretraining_config.yaml'
        OVERRIDE=model=${pairs[$SCRIPT]}
    else
        CONF_ROOT=''
        OVERRIDE=''
    fi
    python ${SCRIPT} --config-path ${prefix}conf ${CONF_ROOT}  -c job --resolve ${OVERRIDE} > "config_tests/${prefix}conf/${FNAME}"
    tail -n +1 "config_tests/${prefix}conf/${FNAME}" > "config_tests/${prefix}conf/${FNAME}.yaml"
    rm "config_tests/${prefix}conf/${FNAME}"
done
done

python compare_configs.py config_tests/conf config_tests/alt_conf
