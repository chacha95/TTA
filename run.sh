#!/bin/bash
cd gpu-0 && docker-compose up -d --build
cd ../gpu-1 && docker-compose up -d --build
cd ../gpu-2 && docker-compose up -d --build
cd ../gpu-3 && docker-compose up -d --build
cd ../gpu-4 && docker-compose up -d --build
cd ../gpu-5 && docker-compose up -d --build
cd ../gpu-6 && docker-compose up -d --build
cd ../gpu-7 && docker-compose up -d --build
