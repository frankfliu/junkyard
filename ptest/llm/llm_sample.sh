#!/bin/bash

# Default values
PROVIDER="gemini"
FEATURE="text"
MODEL=""
USE_VERTEX=false
STREAMING=false

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
  --provider)
    PROVIDER="$2"
    shift
    ;;
  --model)
    MODEL="$2"
    shift
    ;;
  --vertex) USE_VERTEX=true ;;
  --project)
    PROJECT="$2"
    shift
    ;;
  --region)
    REGION="$2"
    shift
    ;;
  --feature)
    FEATURE="$2"
    shift
    ;;
  --streaming) STREAMING=true ;;
  *)
    echo "Unknown parameter passed: $1"
    exit 1
    ;;
  esac
  shift
done

# Read data from data.json using jq
DATA=$(cat "$(dirname "$0")/data.json")

# Determine streaming key
if [ "$STREAMING" = true ]; then
  STREAMING_KEY="streaming"
else
  STREAMING_KEY="non_streaming"
fi

# Get provider data
PROVIDER_DATA=$(echo "$DATA" | jq -r --arg provider "$PROVIDER" '.[$provider]')

# Get model if not provided
if [ -z "$MODEL" ]; then
  MODEL=$(echo "$PROVIDER_DATA" | jq -r '.models[0]')
fi
if [ -z "$PROJECT" ]; then
  PROJECT="\$PROJECT"
fi
if [ -z "$REGION" ]; then
  REGION="global"
fi

# Handle Anthropic model name
if [ "$USE_VERTEX" = false ] && [ "$PROVIDER" = "anthropic" ]; then
  MODEL=${MODEL//@/-}
fi

# Get curl and request templates
CURL_TEMPLATE=$(echo "$PROVIDER_DATA" | jq -r --arg sk "$STREAMING_KEY" '.[$sk].curl')
REQUEST_TEMPLATE=$(echo "$PROVIDER_DATA" | jq -r --arg sk "$STREAMING_KEY" ".[\$sk].${FEATURE}.request")

# Replace model in templates
CURL_TEMPLATE=${CURL_TEMPLATE/\{\{model\}\}//$MODEL}
REQUEST_TEMPLATE=${REQUEST_TEMPLATE//\{\{model\}\}/$MODEL}

# Handle Vertex AI
if [ "$USE_VERTEX" = true ]; then
  URL_TEMPLATE=$(echo "$PROVIDER_DATA" | jq -r --arg sk "$STREAMING_KEY" '.[$sk].url')
  VERTEX_URL=$(echo "$URL_TEMPLATE" | sed "s/{{project}}/$PROJECT/g" | sed "s/{{region}}/$REGION/g" | sed "s/{{model}}/$MODEL/g")

  if [ "$REGION" = "global" ]; then
    VERTEX_URL=$(echo "$VERTEX_URL" | sed 's/https:\/\/global-/https:\/\//')
  fi

  if [ "$PROVIDER" = "anthropic" ]; then
    # shellcheck disable=SC2001
    REQUEST_TEMPLATE=$(echo "$REQUEST_TEMPLATE" | sed 's/\"model\": \"[^\"]*\"/\"anthropic_version\": \"vertex-2023-10-16\"/')
  fi

  if [ "$FEATURE" = "computer_use" ] && [ "$PROVIDER" = "anthropic" ]; then
    CURL_TEMPLATE="curl \"$VERTEX_URL\" \\
    -H \"Authorization: Bearer \$(gcloud auth print-access-token)\" \\
    -H \"Content-Type: application/json\" \\
    -H \"anthropic-beta: computer-use-2025-01-24\" \\
    -d "
  elif [ "$FEATURE" = "memory" ] && [ "$PROVIDER" = "anthropic" ]; then
    CURL_TEMPLATE="curl \"$VERTEX_URL\" \\
    -H \"Authorization: Bearer \$(gcloud auth print-access-token)\" \\
    -H \"Content-Type: application/json\" \\
    -H \"anthropic-beta: context-management-2025-06-27\" \\
    -d "
  else
    CURL_TEMPLATE="curl \"$VERTEX_URL\" \\
    -H \"Authorization: Bearer \$(gcloud auth print-access-token)\" \\
    -H \"Content-Type: application/json\" \\
    -d "
  fi
fi

# Construct final curl command
CURL_COMMAND="$CURL_TEMPLATE'$REQUEST_TEMPLATE'"

# Print the curl command
echo -e "$CURL_COMMAND"
