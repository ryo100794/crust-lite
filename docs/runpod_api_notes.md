# RunPod Pod API notes

更新日: 2026-06-29

## 結論

RunPodのPod起動・停止は、REST API、GraphQL API、`runpodctl` のいずれでも操作できる。
自動化するなら、シンプルな停止/開始はREST API、Pod作成やGPU条件指定まで含めるならGraphQL API、手作業寄りの運用は`runpodctl`が妥当。

## REST API

- Stop: `POST https://rest.runpod.io/v1/pods/{podId}/stop`
- Start/resume: `POST https://rest.runpod.io/v1/pods/{podId}/start`
- 認証: `Authorization: Bearer <token>`

例:

```bash
curl --request POST \
  --url "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/stop" \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}"
```

```bash
curl --request POST \
  --url "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}/start" \
  --header "Authorization: Bearer ${RUNPOD_API_KEY}"
```

## GraphQL API

- Endpoint: `https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}`
- Start/resume mutation: `podResume`
- Stop mutation: `podStop`
- List pods query: `myself { pods { ... } }`
- Get pod query: `pod(input: {podId: "..."})`
- Create pod mutation: `podFindAndDeployOnDemand`
- GPU type list query: `gpuTypes`

Stop:

```bash
curl --request POST \
  --header 'content-type: application/json' \
  --url "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
  --data "{\"query\":\"mutation { podStop(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) { id desiredStatus } }\"}"
```

Start/resume:

```bash
curl --request POST \
  --header 'content-type: application/json' \
  --url "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
  --data "{\"query\":\"mutation { podResume(input: { podId: \\\"${RUNPOD_POD_ID}\\\", gpuCount: 1 }) { id desiredStatus imageName } }\"}"
```

List:

```bash
curl --request POST \
  --header 'content-type: application/json' \
  --url "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
  --data '{"query":"query { myself { pods { id name runtime { uptimeInSeconds gpus { id gpuUtilPercent memoryUtilPercent } container { cpuPercent memoryPercent } } } } }"}'
```

## runpodctl

CLIでも以下が使える。

```bash
runpodctl pod list --all
runpodctl pod get <pod-id>
runpodctl pod start <pod-id>
runpodctl pod stop <pod-id>
runpodctl pod restart <pod-id>
runpodctl pod delete <pod-id>
```

Pod作成時には `--stop-after` と `--terminate-after` がある。前者は停止、後者は恒久削除なので誤用しない。

## 運用方針

- APIキーはGitに入れない。
- 環境変数名は `RUNPOD_API_KEY` と `RUNPOD_POD_ID` に統一する。
- 自動停止スクリプトは、先にGit pushと `/mnt/slam/equake` への成果物同期を実行してから `stop` を呼ぶ。
- `delete` / `terminate-after` は破壊的操作なので、通常運用には入れない。
- 起動時のCPU/GPU割当変更は既存Podのstartだけではなく、Pod作成条件やRunPod側のPod設定確認が必要。

## 参照

- REST Stop a Pod: https://docs.runpod.io/api-reference/pods/POST/pods/podId/stop
- REST Start or resume a Pod: https://docs.runpod.io/api-reference/pods/POST/pods/podId/start
- GraphQL Manage Pods: https://docs.runpod.io/sdks/graphql/manage-pods
- GraphQL Overview: https://docs.runpod.io/sdks/graphql/configurations
- API keys: https://docs.runpod.io/get-started/api-keys
- runpodctl pod: https://docs.runpod.io/runpodctl/reference/runpodctl-pod
