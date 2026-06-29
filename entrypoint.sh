#!/bin/bash
source /roboplan_env.sh

# Mesa writes shader cache to $XDG_CACHE_HOME; /root/.cache isn't writable by the host user.
export XDG_CACHE_HOME=/tmp/.cache
[ -f /usr/share/bash-completion/bash_completion ] && source /usr/share/bash-completion/bash_completion

# Only set prompt for interactive shells (not for `docker run <cmd>`)
if [[ $- == *i* ]]; then
    __git_branch() { git branch --show-current 2>/dev/null | xargs -I{} printf ' [%s]' {}; }
    PS1='\[\033[0;36m\](lerobot-dev)\[\033[0m\] \[\033[0;32m\]\u@\h\[\033[0m\]:\[\033[0;34m\]\w\[\033[0;33m\]$(__git_branch)\[\033[0m\] \$ '
fi

exec "$@"
