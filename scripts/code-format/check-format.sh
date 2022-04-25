#!/bin/sh

SOURCE_COMMIT="$1"
if [ "$#" -gt 0 ]; then
    shift
fi

# If no source commit is given target the default branch
if [ "x$SOURCE_COMMIT" = "x" ]; then
    # If remote is not set use the remote of the current branch or fallback to "origin"
    if [ "x$REMOTE" = "x" ]; then
        BRANCH="$(git rev-parse --abbrev-ref HEAD)"
        REMOTE="$(git config --local --get "branch.$BRANCH.remote" || echo 'origin')"
    fi
    SOURCE_COMMIT="remotes/$REMOTE/HEAD"
fi

# Force colored diff output
DIFF_COLOR_SAVED="$(git config --local --get color.diff)"
if [ "x$DIFF_COLOR_SAVED" != "x" ]; then
    git config --local --replace-all "color.diff" "always"
else
    git config --local --add "color.diff" "always"
fi

GIT_CLANG_FORMAT="${GIT_CLANG_FORMAT:-git-clang-format}"
CLANG_FORMAT_OUTPUT="$($GIT_CLANG_FORMAT --style=file --diff "$@" "$SOURCE_COMMIT")"

# Restore setting
if [ "x$DIFF_COLOR_SAVED" != "x" ]; then
    git config --local --replace-all "color.diff" "$DIFF_COLOR_SAVED"
else
    git config --local --unset "color.diff"
fi

# Check for no-ops
echo "$CLANG_FORMAT_OUTPUT" | grep '^no modified files to format$'           > /dev/null && exit 0
echo "$CLANG_FORMAT_OUTPUT" | grep '^clang-format did not modify any files$' > /dev/null && exit 0

# Dump formatting diff and signal failure
printf \
"\033[31m==== FORMATTING VIOLATIONS DETECTED ====\033[0m
run '\033[33m%s --style=file %s %s'\033[0m to apply these formating changes\n\n" \
"$GIT_CLANG_FORMAT" "$@" "$SOURCE_COMMIT"

echo "$CLANG_FORMAT_OUTPUT"
exit 1
