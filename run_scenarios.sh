#!/bin/bash

# Run all find_days_till_holiday_insufficient_information_3_distraction_tools scenario variants
# with both memory=full and memory=none
# Total: 4 scenarios × 2 memory settings × 10 episodes = 80 runs

cd /Users/chupeng/Documents/Meng_CCDS/task_repeater/ToolSandbox-main

SCENARIOS=(
    "update_contact_with_id_and_phone_number"
    "search_sender_phone_number_with_content"
    # "send_message_with_phone_number_and_content"
    # "add_contact_with_name_and_phone_number"
    # "search_name_with_relationship"
    # "find_days_till_holiday_wifi_off"
    # "modify_contact_with_message_recency"
    # "send_message_with_contact_content_cellular_off"
    # "modify_reminder_with_recency_latest"
    "update_contact_relationship_with_relationship"
    # "update_contact_relationship_with_relationship_twice_multiple_user_turn"
    # "modify_contact_with_message_recency_multiple_user_turn"
    # "find_days_till_holiday_wifi_off_multiple_user_turn"
    # "send_message_with_contact_content_cellular_off_multiple_user_turn"
    # "add_reminder_content_and_weekday_delta_and_time_multiple_user_turn"
    "remove_contact_with_id"
    "search_relationship_with_phone_number"
    # "search_phone_number_with_name"
    # "find_thanksgiving_timestamp"
    # "cellular_off"
    # "remove_reminder_with_recency_latest"
    "search_reminder_with_creation_recency_yesterday_implicit"
    # "search_reminder_with_recency_yesterday_implicit"
    "turn_on_wifi_low_battery_mode_implicit"
    "add_reminder_content_and_weekday_delta_and_time"
    # "search_message_with_recency_oldest_multiple_user_turn"
    # "remove_contact_by_phone_multiple_user_turn"
    # "update_contact_relationship_with_relationship_multiple_user_turn"
    # "find_days_till_holiday_multiple_user_turn"
    # "add_reminder_content_and_week_delta_and_time_multiple_user_turn"
    # "modify_reminder_with_recency_latest_insufficient_information"
    # "remove_reminder_with_recency_latest_insufficient_information"
    # "modify_contact_with_message_recency_insufficient_information"
    # "send_message_with_contact_content_cellular_off_insufficient_information"
    # "search_reminder_with_creation_recency_yesterday_insufficient_information_implicit"
    # "search_reminder_with_recency_yesterday_insufficient_information_implicit"
    # "search_reminder_with_recency_upcoming_insufficient_information_implicit"
    # "find_days_till_holiday_insufficient_information"
    "remove_contact_by_phone_no_search_contacts_insufficient_information"
    "remove_contact_by_phone_no_remove_contact_insufficient_information"
)

SUFFIX=(
    # "_3_distraction_tools"
    # "_3_distraction_tools_arg_description_scrambled"
    # "_3_distraction_tools_arg_type_scrambled"
    # "_3_distraction_tools_tool_description_scrambled"
    "_10_distraction_tools"
    # "_10_distraction_tools_arg_description_scrambled"
    # "_10_distraction_tools_arg_type_scrambled"
    "_10_distraction_tools_tool_description_scrambled"
)

AGENT="DeepSeek_V4_Flash_0731"
USER="end"
EPISODES="10"
SEED="42"

# Create output directory
# mkdir -p /Users/chupeng/Documents/Meng_CCDS/task_repeater/ToolSandbox-main/experiment_results

for scenario in "${SCENARIOS[@]}"; do
    for suffix in "${SUFFIX[@]}"; do
        for memory in "full" "none"; do
            echo "Starting: $scenario with memory=$memory"
            # output_dir="/Users/chupeng/Documents/Meng_CCDS/task_repeater/ToolSandbox-main/experiment_results/${scenario}_memory_${memory}"

            .venv/bin/tool_sandbox \
                --agent "$AGENT" \
                --user "$USER" \
                -s "$scenario$suffix" \
                --episodes "$EPISODES" \
                --memory "$memory" \
                --episode-parameter-seed "$SEED" \
                # --output_dir "$output_dir" &

            # Add a small delay between starting processes
            sleep 2
        done
    done
done

echo "All 80 runs have been submitted to background. Check the experiment_results directory for outputs."
