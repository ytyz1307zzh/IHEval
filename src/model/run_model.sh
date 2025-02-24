# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# Make sure to add OpenAI API key to the environment variable OPENAI_API_KEY

# Change this to specify which category(domain) of tasks to run
domains=($(find benchmark -mindepth 1 -maxdepth 1 -type d | xargs -n 1 basename))  # run all categories
# domains=(tool-use)  # only run tool-use tasks

# Change this to modify which model to run
model_list=(llama3.1-8b llama3.1-70b mistral-7b gpt-4o-mini-2024-07-18)

for domain in ${domains[@]}; do

    # Change this to specify which task to run
    system_tasks=($(find benchmark/${domain} -mindepth 1 -maxdepth 1 -type d | xargs -n 1 basename))  # run all tasks
    # system_tasks=(slack-user)

    for system_task in ${system_tasks[@]}; do

        for model in ${model_list[@]}; do
            
            if [ "$model" = "llama3.1-8b" ]; then
                model_family=llama
                model_path=meta-llama/Llama-3.1-8B-Instruct
                script_type=vllm
            elif [ "$model" = "llama3.1-70b" ]; then
                model_family=llama
                model_path=meta-llama/Llama-3.1-70B-Instruct
                script_type=vllm
            elif [ "$model" = "mistral-7b" ]; then
                model_family=mistral
                model_path=mistralai/Mistral-7B-Instruct-v0.3
                script_type=vllm
            elif [[ "$model" == gpt* ]]; then
                model_family=gpt
                model_path=${model}
                script_type=api
            else
                echo "Model not found"
                exit 1
            fi

            # Find all data directories for the given task
            folders=($(find benchmark/${domain}/${system_task} -mindepth 2 -maxdepth 2 -type d | awk -F '/' '{print $(NF-1)"/"$NF}'))
            # If only evaluating specific settings, e.g., the aligned setting, change to folders=(aligned/default)

            for folder in ${folders[@]}; do

                echo -e "\n\e[31m********************** ${domain}/${system_task}/${folder} **********************\e[0m"

                OUTPUT_DIR=benchmark/${domain}/${system_task}/${folder}/${model_family}/${model}

                # For vllm only: tensor_parallel
                # For API call only: max_retries, max_threads, sleep
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

                echo -e "\e[32m--------------------- Process Scores ---------------------\e[0m"

                # For tool use, we may need to calculate the aggregated score of three NLP tasks
                if [ "$system_task" = "get-webpage" ] ; then
                    if [ "$folder" = "reference/default" ] ; then
                        python src/task_execution/evaluate/calc_mix_reference_score.py \
                            -input benchmark/${domain}/${system_task}/${folder}/${model_family}/${model}/eval_results.json \
                            -record_dir model-scores
                    else
                        python src/task_execution/evaluate/calc_mix_task_score.py \
                            -input benchmark/${domain}/${system_task}/${folder}/${model_family}/${model}/eval_results.json

                        python src/model/record_scores.py \
                            -data benchmark/${domain}/${system_task}/${folder}/${model_family}/${model}/eval_results.json \
                            -output_dir model-scores
                    fi

                # For calculating the reference score of NLP tasks (except for language detection task)
                elif [ "$domain" = "task-execution" ] && [ "$folder" = "reference/default" ] && [ "$system_task" != "lang-detect" ]; then
                    python src/task_execution/evaluate/calc_reference_score.py \
                        -input benchmark/${domain}/${system_task}/${folder}/${model_family}/${model}/eval_results.json \
                        -task ${system_task} \
                        -record_dir model-scores
                
                # For recording the model scores
                else
                    python src/model/record_scores.py \
                        -data benchmark/${domain}/${system_task}/${folder}/${model_family}/${model}/eval_results.json \
                        -output_dir model-scores
                fi

            done

            # Aggregate the scores of all the tasks for a single model
            python src/model/average_final_score.py \
                -record model-scores/${model}.json \
                -output model-scores/overall/overall_${model}.json

        done
    
    done

done
