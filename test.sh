#!/bin/bash

source env/bin/activate

ruff check
pyright
