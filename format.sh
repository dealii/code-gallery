#!/bin/bash

find . -name '*.h' -o -name '*.cc' -print0 |
        xargs -0 -n 1 -P 10 -I {} bash -c "clang-format -i {}"
