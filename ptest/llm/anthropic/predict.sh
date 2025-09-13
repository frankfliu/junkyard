#!/usr/bin/env bash

MODEL=claude-3-5-haiku@20241022

# Pick one region:
LOCATION=us-east5

#time curl -X POST \
#  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
#  -H "Content-Type: application/json; charset=utf-8" \
#  -d @request.json \
#  "https://$LOCATION-aiplatform.googleapis.com/v1/projects/$PROJECT/locations/$LOCATION/publishers/anthropic/models/$MODEL:rawPredict"

#  "https://$LOCATION-aiplatform.googleapis.com/v1/projects/$PROJECT/locations/$LOCATION/publishers/anthropic/models/$MODEL:streamRawPredict"

time curl -X POST \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json; charset=utf-8" \
  -d @pdf.json \
  "https://$LOCATION-aiplatform.googleapis.com/v1/projects/$PROJECT/locations/$LOCATION/publishers/anthropic/models/$MODEL:rawPredict"
