# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# Make sure to add OpenAI API key to the environment variable OPENAI_API_KEY

domains=($(find benchmark -mindepth 1 -maxdepth 1 -type d))
system_tasks=($(find benchmark/${domain} -mindepth 1 -maxdepth 1 -type d))

model_list=(gpt-3.5-turbo gpt-4o-2024-08-06 llama3-8b llama3-70b claude3-haiku claude3-sonnet)

for model in ${model_list[@]}; do
    
    if [ "$model" = "llama3-8b" ]; then
        model_family=llama
        model_path=meta.llama3-8b-instruct-v1:0
        script_type=api
    elif [ "$model" = "llama3-70b" ]; then
        model_family=llama
        model_path=meta.llama3-70b-instruct-v1:0
        script_type=api
    elif [ "$model" = "qwen2-7b" ]; then
        model_family=qwen
        model_path=Qwen/Qwen2-7B-Instruct
        script_type=lmdeploy
    elif [ "$model" = "qwen2-72b" ]; then
        model_family=qwen
        model_path=Qwen/Qwen2-72B-Instruct
        script_type=lmdeploy
    elif [ "$model" = "claude3-sonnet" ]; then
        model_family=claude
        model_path=anthropic.claude-3-sonnet-20240229-v1:0
        script_type=api
    elif [ "$model" = "claude3-haiku" ]; then
        model_family=claude
        model_path=anthropic.claude-3-haiku-20240307-v1:0
        script_type=api
    elif [[ "$model" == gpt* ]]; then
        model_family=gpt
        model_path=${model}
        script_type=api
    else
        echo "Model not found"
        exit 1
    fi

    # Find all data directories for the given task
    folders=($(find benchmark/${domain}/${system_task} -mindepth 2 -maxdepth 2 -type d))

    for folder in ${folders[@]}; do
        # Remove the leading "./" from the folder path
        folder=${folder#./}

        echo "********* ${folder} *********"

        OUTPUT_DIR=benchmark/${domain}/${system_task}/${folder}/${model_family}/${model}

        python src/model/run_model.py \
            -model ${model_path} \
            -input benchmark/${domain}/${system_task}/${folder}/input_data.json \
            -request_file ${OUTPUT_DIR}/input_request.json \
            -response_file ${OUTPUT_DIR}/input_response.json \
            -eval_output_dir ${OUTPUT_DIR} \
            -max_tokens 2048 \
            -temperature 0.0 \
            -top_p 1.0 \
            -top_k 1 \
            -precision auto \
            -task ${system_task} \
            -backend ${script_type} \
            -max_retries 10 \
            -max_threads 16 \
            -sleep 5

        # For tool use, we may need to calculate the aggregated score of three nlp tasks
        if [ "$system_task" = "mixed-nlp-tasks" ]; then
            python src/nlp_task/data/calc_mixed_score.py \
                -input benchmark/${domain}/${system_task}/${folder}/${model_family}/${model}/eval_results.json
        fi

        # For calculating the reference score
        if [ "$domain" = "task-execution"] && [ "$folder" = "reference" ]; then
            python src/task_execution/data/calc_reference_score.py \
                -input benchmark/${domain}/${system_task}/${folder}/${model_family}/${model}/eval_results.json \
                -task ${system_task}
        fi

    done

done
