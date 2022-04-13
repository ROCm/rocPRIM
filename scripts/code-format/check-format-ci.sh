#!/bin/sh

cd $CI_PROJECT_DIR
git config --global --add safe.directory $CI_PROJECT_DIR

echo "running $GIT_CLANG_FORMAT --style=file --diff $CI_MERGE_REQUEST_DIFF_BASE_SHA"
CLANG_FORMAT_OUTPUT="$($GIT_CLANG_FORMAT --style=file --diff "$CI_MERGE_REQUEST_DIFF_BASE_SHA")"

# Check for no-ops
echo "$CLANG_FORMAT_OUTPUT" | grep '^no modified files to format$'           > /dev/null && exit 0
echo "$CLANG_FORMAT_OUTPUT" | grep '^clang-format did not modify any files$' > /dev/null && exit 0

# Dump formatting diff and signal failure
printf "\033[31m==== FORMATTING VIOLATIONS DETECTED ====\033[0m"

echo "$CLANG_FORMAT_OUTPUT"
exit 1
