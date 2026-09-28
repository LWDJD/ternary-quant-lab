# read-only-git（dev 分支，c2f6247）对抗测试报告

测试日期：2026-09-24 晚。测试方式三层并行：
1. **基线**：`go test ./...` 全绿；`node scripts/selftest.mjs`（修复 import 路径后）全绿
2. **实测对抗**：真实二进制 + 真实服务 + 恶意输入打到底，附复现命令
3. **代码审查**：四条攻击面并行审查（CLI/repopack/publish、arweave、webui/signer、前端）

环境说明：Windows 沙箱内 `git clone` 本地传输受限（README 已知的 signal pipe 场景），恰好完整验证了 bundle 回退链路；浅克隆路径未能在本环境覆盖。实测产物在 `%TEMP%\rog-adv-target\`。

---

## 严重度说明

- **高**：可被直接利用，破坏数据或打挂访客
- **中高**：利用有前置条件（同机写权限/网关被控），但后果真实
- **中**：门禁缺口、竞态、语义违背
- **低**：功能缺陷、体验/健壮性

验证标记：【实测】亲手复现；【代码】代码级确认机制成立，未跑通完整利用链；【子审】子实例审查结论，未独立复核。

---

## 一、高

### A1【实测】webui 请求体 `site` 字段任意目录逃逸 —— 任意文件读写删

- 位置：`internal/webui/api.go:77-82` `siteOf()`：`given` 非空则**原样当根目录**，零校验。`safeJoin`（api.go:53）只约束相对路径，根被请求方任意指定
- 影响接口：`/api/state`（含 GET query `site`）、`/api/files/replace|delete|copy|mkdir`、`/api/site/init`、`/api/pack|publish|restore` 全部走 `siteOf(s, req.Site)`。同族独立成洞的参数还有：`/api/pack` 的 `outDir`、`/api/publish` 的 `dest`、`/api/restore` 的 `dest`（子审 S5），restore 尤甚，与 A7 合体后见 A12
- 后果：持 token 的请求可枚举任意目录（含每个文件的 sha256）、在任意位置建目录/删文件/拷文件/写文件。设计承诺是「token → 只能改站点文件」，实际是「token → 全盘」
- 复现（安全目标验证）：
  ```
  POST /api/files/mkdir   {"site":"D:\\...\\read-only-git-test","path":"escape-proof-dir"}   → 200，目录出现在站点根之外
  GET  /api/state?site=D:\...\read-only-git-test                                              → 200，列出任意目录文件清单+摘要
  POST /api/files/delete  {"site":"D:\\...\\read-only-git-test","path":"escape-proof-dir"}   → 200，清理成功
  ```
- 前置条件：会话 token。但 token 一旦经 XSS/Referer 泄漏（见 A9 缓解项）或用户被社工，打击面从站点目录升到整机文件系统
- 修复：`siteOf` 只允许 `s.siteDir`（或对 `given` 做白名单 + `filepath.Rel` 包含性复核）；所有路径操作统一三段式校验

### A2【实测】浏览器端 git 解析器三条 DoS —— 恶意仓库打挂访客标签页

威胁模型：站点浏览者打开恶意仓库页面即触发（HEAD 树指向恶意对象即可）。payload 都是几十 KB 以内。

- **A2-1 delta 自指/成环死循环**：`internal/sitekit/site/src/git/pack.js:83-88` `ofs_delta` 递归 `readAt(baseOffset)` 无环检测、无深度限制。`baseOffset=0` 即指向自身；ref_delta 双对象互指同理
  - 实测：~30 字节 pack，5 秒内存 42MB→269MB 且永不结束（`pack-adv.mjs` case1，看门狗超时）
  - 修复：递归改迭代 + visited(offset) 集合 + 深度上限
- **A2-2 delta 分配+触页炸弹**：`delta.js:19` `new Uint8Array(resultSize)` 直接信任 delta 头声明值，无上限
  - 实测：约 20KB pack（64KiB base + 16384 条 copy 指令），`resultSize=2^30` → RSS +1GB 并成功返回；`2^32-1` → RSS 冲到 4.2GB 后 RangeError（`pack-adv.mjs` case2）
  - 修复：resultSize 上限（如 512MB）+ 与实际产出增量校验
- **A2-3 解压炸弹**：`zlib.js:41-53` `collectAll` 无膨胀上限，且在 `pack.js:76` 的 size 校验**之前**就完成膨胀；`concat` 再复制一份，放大 2 倍
  - 实测：254.8KiB 压缩体膨胀到 256MB（RSS +620MB），随后才抛 size mismatch（`pack-adv.mjs` case3）。线性外推 1MB payload 可打 1GB+
  - 修复：流式解压时计数截断，超过声明 size 即中止

### A3【实测】pack 增量分支经 junction/symlink 写穿站点外仓库（删 ref）

- 位置：`internal/repopack/repopack.go:169` 增量判定 `isUsableTarget(target)` 直接作用于 `filepath.Join(outRoot, name)`，不拒绝符号链接；增量路径 `fetchInto`/`syncRefs`（update-ref -d）/`prune` 全部原地写目标
- 后果：`站点目录/<name>` 是指向别处裸仓库的 junction 时（攻击者对站点根有写权限即可预置），受害者默认参数 `rog pack <src> <site> <name>` 会把站点外那个仓库当增量目标：ref 被对齐/删除，`prune` 清掉 hooks 等文件
- 实测（现实形态：victim 与源同历史的镜像 + 独有 tag `victim-only`）：
  ```
  > 增量更新已有仓库
  > 增量完成，新增 0 / 更新 0 / 删除 1 个 ref，当前 pack 1 个
  ```
  victim 的 `refs/tags/victim-only` 被隔着 junction 删除。完全无关仓库会在 `bundle create --not <sha>` 阶段因 bad object 中止（未写穿，但机制已暴露）
- 附带观察：junction 目标的文件清单渲染异常（`demo/.` 4.0 KiB，总计 0 B）
- 修复：增量前对 `target` 及各级父目录 `Lstat`，遇 `ModeSymlink`/reparse point 拒绝

---

## 二、中高

### A4【代码】publish 本地拷贝跟随符号链接，可任意文件覆盖

- 位置：`internal/publish/local.go:50,95,110`：`MkdirAll` 与 `OpenFile(O_TRUNC)` 默认跟随符号链接。目标目录预置 `dest/index.html -> ~/.ssh/authorized_keys` 类链接（或父级 junction），`rog publish site dest` 写穿覆盖任意文件
- 修复：写前 `Lstat(dst)` 拒绝链接，逐级组件检查 reparse point

### A5【实测】repository.json 解析失败静默清空全部条目 + 非原子写 + 锁粒度按仓库名

- 位置：`internal/repopack/repopack.go:825,849,885`；锁 `acquireLock(outRoot, name)`
- 三个问题叠加：解析错误当空清单 → 只写当前条目（**其余仓库登记全部丢失**）；`os.WriteFile` 截断重写非原子，写一半被杀即产生非法 JSON 触发上一条；锁按仓库名分，不同名并发 pack 同一站点根互不互斥 → repository.json 丢失更新
- 实测：pack `one` 后把 repository.json 改成 `{"repositories": {}}`，再 pack `two` → 结果只剩 `two`，`one` 消失
- 修复：解析失败即中止报错；临时文件 + rename 原子写；锁键改为站点根

### A6【代码】链上发布记录可投毒（`--from` 信任链）

- 位置：`internal/arweave/target.go:104-110` 复用判定只信 `prev.Refs[path]` 字符串，从不校验该 id 内容；`main.go:386-402` 对 `--from` 取回的记录直接 `SaveRecord` 并当 `prev`
- 利用：攻击者把公开可算的 `Files` 摘要照抄、`Refs` 换成自己的数据 id，发布伪造记录；诱使受害者 `--from <攻击者入口>`（或 `--gateway` 指向恶意网关）续增量 → 新 manifest 把路径指向攻击者内容，且伪造记录覆盖本地 `.rog/` 后长期生效
- 修复：复用前按 id 取回内容核验摘要，或记录自带签名

### A7【代码】从链上恢复全链路无内容认证

- 位置：`internal/arweave/restore.go:29-66,74-138,168-188`：manifest 与文件字节都直接信网关 `/raw` 响应，无 owner/签名校验（Arweave id 是签名哈希不是内容哈希，无法按 id 验内容）
- 利用：恶意网关（或 `--gateway` 被诱导指向恶意地址）返回任意 manifest + 任意文件字节，恢复即写盘；用户随后再发布等于替攻击者发布。路径侧 `safeRestorePath`（restore.go:151-157）挡住了 `..`，但比 webui 的 `safeJoin` 少两道兜底（无 IsAbs 拒绝、无 Rel 复核，绝对路径被静默相对化）
- 修复：取完整签名数据项校验 owner 与签名；safeRestorePath 与 safeJoin 同源

---

## 三、中

### A8【代码】`IsRemote` 接受前导 `-` 的 scp 串 → git 选项走私

- 位置：`internal/repopack/repopack.go:69` `remoteSCPRe` 字符类含 `-`；`:203` `git clone --bare --quiet <source> <target>` 无 `--` 分隔（同 CVE-2017-1000117 形态）
- 武器化上限：含 `=`、`/`、空格的经典 payload 不匹配 scp 正则，落不进远端分支；实际后果为选项解析混乱/DoS，RCE 不可达。修复便宜：IsRemote 拒绝前导 `-`，或 argv 插 `--`

### A9【代码】本地发布无跨进程锁；CLI 根本没接进程内锁

- 位置：`internal/publish/lock.go:27,53` 仅进程内 map；`main.go:287` `cmdPublishLocal` 未调用。并发 `rog publish` 时目标文件 O_TRUNC 原地写，外部读者可见半截文件。另 lockKey 不含 dest 维度（不同仓库互相挡、同目标不隔离）

### A10【代码】签名回传 `uploaded` 免检；本地验签不绑定 bundle

- 位置：`internal/arweave/target.go:275-291`（`uploaded==true` 直接记成功，不验签不确认）、`verify.go:30-110`（验「页面回传字段自洽」，`data_root` 与本地 bundle 无关联，Go 从不重算 Merkle）
- 免检场景需控制签名页/本机服务，超出外部敌手模型，但「页面说传好了=发布成功」的门禁低于直觉

### A11【代码】Turbo 上传响应 id 无校验直入 manifest

- 位置：`internal/arweave/uploader.go:128-158`、`target.go:99-113`。`--endpoint` 被控/MITM 时 manifest 可被指向攻击者内容

---

## 四、低（功能与健壮性）

### F1【实测】`pnpm selftest` 开箱即坏

- `scripts/selftest.mjs:18` import `../public/src/git/repo.js`，但 f0ee61c 之后前端真身在 `internal/sitekit/site/src/`，`public/` 是产物目录（干净克隆下不存在）。README:315 却把它写成开箱即用。`rog site init public` 后可跑通
- 修复：selftest 改指 `internal/sitekit/site/src/git/repo.js`

### F2【实测】pack 帮助/文档语义与实现不符

- README 与 `rog pack --update`：**`--update` 不存在**，被当源仓库路径（报「源仓库不存在: --update」）。实际旗标是 `--rebuild/--full/-r`（main.go:649）
- README「重复执行安全：同名仓库覆盖重建」与实际相反：实现是**默认自动增量**（repopack.go:169 `incremental := !opt.Rebuild && ...`），同名重跑保留旧 pack
- 修复：文档跟实现对齐，或恢复 `--update` 为显式开关

### F3【实测】Windows 名称校验缺口与同目录碰撞

- `isValidName` 放行尾点名 `x.`（进流程后 git 报原始错误 `cannot mkdir ...: Invalid argument`）；`CONIN$`/`CONOUT$` 同样漏过校验在 git 层炸原始错误；`COM¹/COM²/COM³`（上标数字映射 COM1-3 设备名）被完整打包成功，留设备名混淆隐患
- 尾点/尾空格名与去空格名在 Win32 同目录等价：`x.`/`x `/`x` 指向同一目录，叠加「同名覆盖重建」语义可静默互相覆盖（`x.` 在已有 `x` 时对其执行了 `git init --bare` 重初始化）
- 锁文件按 name 命名（`.rog/<name>.lock`），`x` 与 `x.` 共锁、与 `x2` 不共锁，语义错位
- 修复：拒绝尾随点/空格、补 CONIN$/CONOUT$/上标设备名黑名单；NormalizeName 后做一次 Win32 归一化碰撞检测

### F4【实测】杂项

- 仓库名 `-x`、`--help` 被接受为目录名（本地无害，URL/展示层面怪异）
- 空名或全空格名静默回退「源目录名」，`rog pack src site "work "` 与 `rog pack src site work` 同目标（文档未写 trim 语义）
- junction 目标的 pack 文件清单渲染异常（`demo/.` 4.0 KiB，总计 0 B）
- webui `/api/state` 响应泄漏服务端 cwd、logDir 路径（信息量小）
- token 比较用 `==` 非常量时间（128bit 随机 + 本机回环，实际难利用，建议顺手换 `subtle.ConstantTimeCompare`）

### F5【代码】arweave 杂项（子审，未独立复核）

- `restore.go:181`、`fetch.go:46` `LimitReader` 到顶静默截断，64MiB 档文件被截断仍写盘
- `restore.go:169`、`fetch.go:37` id/entry 未转义拼 URL 路径（`..`、`?`、`#` 可改请求路径，影响限于网关主机）
- `chunk.go` `MinChunkSize` 死代码，分块大小不校验（恶意签名页可造请求风暴；内容错位不可能）
- restore 中途失败留半成品目录；重试语义遗漏 408 等

### F6【子审】CLI 边界杂项

- **`--rebuild` 配合「源在输出目录内」会先删源**：`repopack.go:187` `removeDirRetry(target)` 在全量分支先删目标；若源就住在目标里（如 `rog pack out/demo out demo --rebuild`），源先被删光，随后打包报错。cmdPack 不校验二者关系。危害限于自伤，但属真实数据丢失路径，低危首位
- **`.rog` 为链接时状态写入被重定向**：`repopack.go:708`、`publish.go:244` 的 `MkdirAll(.rog)` 对链接 Stat 后判目录通过，锁文件与发布记录落进链接目标
- **cmdPack 未知开关被当位置参数**：解析循环只识别 `--rebuild/--full/-r/--proxy`，`--prox` 这类错写会被当 OutDir，报错指向无关位置；cmdPublish 对未知开关会明确报错，两者行为不一致
- **发布记录 identity 用未规范化的 destDir**：`publish site out` 与 `publish site ./out` 生成两个记录文件，增量复用被白丢；identity 哈希仅取前 4 字节，建议 8+
- **锁键碰撞可绕锁**：`foo`/`foo.`/`Foo` 在 Win32 同目录但锁文件名不同（`foo.lock`/`foo..lock`），互不互斥（与 F3 同根，并发时序下可互踩中间态）

### I1【子审】file:// 源触发 git 侧配置执行（前提级）

- `repopack.go:68` 把 `file://` 归为远端走 smart 传输，本地 `git-upload-pack` 在**源仓库自己的 config** 下运行：恶意源仓库可在 config 里写 `uploadpack.packObjectsHook`，pack 它即执行其中命令
- 不是 rog 拼 shell 造成的，是「把不可信仓库当数据」这一前提被 git 语义打破。建议文档显式声明「不要 pack 不可信来源」，或硬化配置（`-c uploadpack.packObjectsHook=` 等）覆盖之

---

## 四·补 webui/signer 深查补遗（A12~A21）

以下来自补遗审查（子审，行号精确、代码事实，未逐条独立复跑；与前文交叉引用）。token 前置的条目在 S10 的 Referer 链条成立时全部解锁，构成一条完整攻击链。

### A12【高】webui restore：远端内容直落任意目录（A1+A7 合体升级）

- 位置：`internal/webui/api.go:605` `handleRestore`；`internal/arweave/restore.go:169-170,118`。`gateway` 与 `dest` 全程无校验：gateway 拼进 `/raw/<id>` 取内容，dest 经 `filepath.Abs` 后 `MkdirAll`，`safeRestorePath` 只挡 `../`，不限制绝对目标（目标根本身就是自由的）
- 后果：持 token 一方起一台返回合法 manifest 的恶意服务器，即可把任选内容写到任选目录（如 Windows 启动文件夹）→ 直通代码执行。比 A1 更重：内容来自网络，不依赖本机已有文件
- 复现配方：本地 HTTP 服务回 `{"manifest":"arweave/paths","paths":{...}}` + 各 id 任意字节；`POST /api/restore {"entry":"x","dest":"<任意绝对目录>","gateway":"http://127.0.0.1:8xxx"}`

### A13【高】gateway/endpoint/node/from 无主机校验：SSRF + 签名字节外送

- 位置：`api.go:350`（Endpoint/Node/Gateway/From）、`api.go:605`（Gateway）；落点 `arweave/fetch.go:35-38`、`uploader.go:96-99`、`l1.go:231-233`、`node.go:52`。仅 TrimRight 去尾斜杠后拼 URL，无 scheme/host 白名单、无内网拦截
- 后果：向内网任意地址 GET（from/restore）或 POST 签好名的 data item 字节（turbo/l1）；`pack` 的 `ssh://`/scp 源还会拉起 ssh 出站

### A14【高】signer：blob 与 sign 不绑定，签名结果无条件采信（A10 攻击面补全）

- 位置：`internal/signer/service.go:309-341`（handleSign 的 :322 `io.ReadAll` 与 :337 `it.result <- body`）、:298 handleBlob、:222 enqueue。POST body 原样当签名结果送回，从不校验与 `it.data`/`it.tags` 的对应关系；谁先 POST 谁赢
- 后果（kind=tx）：对已知 id POST `{"uploaded":true,...}` → `submitL1` 免检判成功、ClearPending、整轮 publish 报「入口 <root>」而链上什么都没发（叠加 A10）。kind=dataitem 时可顶替真实内容上传，且 `/api/blob` 把待上链内容吐给任何持 token 者（发布前泄露）
- 修复：结果与 it.data 绑定校验；Uploaded:true 必须本地 VerifySignedTx 后才采信

### A15【中】所有 POST 端点无 body 上限

- `api.go:38` decodeBody 的 `json.NewDecoder(r.Body).Decode` 与 `signer/service.go:322` `io.ReadAll(r.Body)` 均无 MaxBytesReader/Content-Length 校验；`/api/files/replace` 的 base64 bytes、`/api/task/<id>/note` 同理。单请求拉爆进程内存

### A16【中】copyTree/写文件跟随软链接，循环检测是词法的（A4 同族）

- `api.go:865` copyTree（os.Stat 跟随链接、递归无深度上限）、:851 inside（只做 Abs 不解析链接）、:710 writeSiteFile（WriteFile 跟随）。站点内预置外向 junction 可把外部整棵拉进站点（读取放大）；指向祖先的链接让 copyTree 无限递归，inside 拦不住

### A17【中】任务 id 可枚举，/note 日志注入，任务结束后仍可写

- `task.go` id 形如 `publish-1` 可预测；`GET /api/task/<id>` 泄任意任务日志；`handleTaskNote` 的 `TrimSpace` 只去首尾，中间 `\n` 原样入日志文件 → 伪造日志行；结束后 sink 已关但 `t.logs` 内存切片仍无上限增长

### A18【中】signer 队列永不回收，发布期间整站内容常驻内存

- `signer/service.go:232-233` items/order 只增不删，每个待签项存完整 data []byte；大站点发布内存累积，叠加 A15 加速 OOM

### A19【中低】安全响应头缺失 + token 驻 URL（攻击链解锁件）

- `server.go:136-139` 无 CSP/X-Frame-Options/Referrer-Policy/nosniff；`page.go:285` token 取自 location.search，:1218 EventSource 拼进 query。现代浏览器默认 strict-origin-when-cross-origin 下跨域不带 query，但无显式 Referrer-Policy 时防线全押浏览器默认值；一旦 token 经 Referer/历史/截图泄露，A12/A13/A14 全部解锁。另无 frame-ancestors，可点击劫持诱导点「删除/发布」

### A20【低】SSE 只转义 \n 不处理 \r

- `server.go:309` escapeSSE 仅 ReplaceAll `\n`；SSE 规范里 `\r` 也是行终止符，可伪造日志流行或提前 end。页面 textContent 渲染（与前文 XSS 结论一致），非 XSS，仅日志伪造

### A21【低】无 Host 校验（DNS rebinding）+ gitProxy 包级变量竞态

- 敏感路由全有 token 可挡住 rebinding；但 `repopack.gitProxy` 是包级变量，锁粒度 `(outRoot,name)` 不覆盖全局代理设置，并发 pack 时代理串味

### 补遗确认项（与前文结论互证）

- 日志行渲染 textContent，无 XSS sink；handleBlob/handleSign 的 id 是 map 查表，无路径拼接；handleTask 路由查表无穿越
- webui 不经 main.go 的 flag 解析（直接调 repopack/publish/arweave），CLI 旗标走私在 webui 路径上不适用；前导 `-` 的源路径注入被 dirExists/URL scheme 判定拦得住。真正的洞是目录无约束（A1/A12/A5），不是旗标

---

## 五、判定为干净的面（附理由）

- **shell 注入**：全仓无 `sh -c`/`cmd /c`，git 调用全部 `exec.Command("git", args...)`，分号/反引号/换行注入不成立
- **仓库名注入 repository.json**：`isValidName` 挡控制符/分隔符/`<>:"|?*`，序列化走 `json.Encoder`
- **webui 相对路径穿越**：`safeJoin` 三段式（拒前导分隔符 + IsAbs + Rel 复核）经实测 `../../`、`..\..\`、绝对路径、UNC 全部拒绝
- **webui/signer token**：两层 guard、header/query 双通道，无 token/错 token 实测 403；signer 端点独立 guard 生效
- **XSS**：markdown.js 入口 escapeHtml + 25 个载荷（raw HTML、javascript:/data: URL、实体二次注入、属性逃逸）实测全转义；page.go、signer 页无 innerHTML 类 sink；views.js 唯一 innerHTML 喂的是 markdown 输出
- **deephash/verify 字段覆盖**：九项编码对照 arweave-js v2 规范，重放被 data_root/last_tx 绑死，canonicalInt 拒非规范整数
- **chunk 分块数学**：边界由 offset 推出并强校验精确覆盖，多组边界 case 与 arweave-js 对拍通过；pending 复用有 SameBundle + 复核验签
- **dumb HTTP 端到端**：静态服务器 + 双 pack 仓库被真实 `git clone` 完整拉取，refs/内容一致（含 Range 请求路径）

---

## 六、未覆盖项（诚实边界）

- 浅克隆拒绝路径（沙箱禁 git 本地传输）
- Turbo / L1 真链发布与签名页真实钱包流（需要网络与钱包，代码审查替代）
- 跨进程并发锁竞态（只做了机制分析；A21 的 gitProxy 竞态为代码事实）

## 复现工具

- `xss-probe.mjs`：markdown XSS 载荷集（`node xss-probe.mjs`）
- `pack-adv.mjs`：恶意 pack/idx 生成器 + 解析器 DoS 三案例（`node pack-adv.mjs`）
- 实测靶场：%TEMP%\rog-adv-target\（adv1~adv8 各轮站点）
