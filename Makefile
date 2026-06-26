IMAGE = roboplan_lerobot:latest
UID  := $(shell id -u)
GID  := $(shell id -g)

.PHONY: build run test clean

build:
	docker compose build

run: _ensure_built
	UID=$(UID) GID=$(GID) SSH_AUTH_SOCK=$(SSH_AUTH_SOCK) docker compose run --rm roboplan_lerobot bash

test: _ensure_built
	UID=$(UID) GID=$(GID) SSH_AUTH_SOCK=$(SSH_AUTH_SOCK) docker compose run --rm roboplan_lerobot python3 -c "\
import roboplan.core; print('roboplan  ok'); \
import mujoco;        print('mujoco   ', mujoco.__version__); \
import lerobot;       print('lerobot   ok'); \
import torch;         print('torch    ', torch.__version__); \
print('cuda:', torch.cuda.is_available()); \
"

clean:
	docker rmi $(IMAGE) || true

_ensure_built:
	@docker image inspect $(IMAGE) > /dev/null 2>&1 || $(MAKE) build
