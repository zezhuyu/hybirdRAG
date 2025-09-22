#! /bin/bash

python -m grpc_tools.protoc --python_out=. --grpc_python_out=. --proto_path . ml_models.proto