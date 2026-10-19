<template>
  <h1>研究备忘录</h1>
  <div class="sub">DeepAgents：研究主管列计划 → 派财务分析师、估值分析师、年报研究员、行业研究员 → 汇总成带证据编号和年报页码的备忘录 + 带公式的 Excel DCF 模型</div>
  <div class="card row">
    <input v-model="ticker" placeholder="000333 / 00700 / AAPL" style="width:180px" @keyup.enter="start" />
    <label class="muted"><input type="checkbox" v-model="review" /> 估值假设需人工审批</label>
    <button :disabled="running" @click="start"><span v-if="running" class="spin"></span>{{ running ? `研究中… ${elapsed}s` : '开始研究' }}</button>
    <span class="muted">支持 A 股、港股、美股；A 股首次研究会先入库年报（约 1 分钟）；A 股银行、保险、券商自动改用 PB-ROE / DDM</span>
    <span class="bad" v-if="error">{{ error }}</span>
  </div>

  <div class="grid g31">
    <div>
      <div class="card" v-if="events.length">
        <h3>Agent 工作过程</h3>
        <div class="timeline" ref="tl"><div v-for="(e, i) in events" :key="i"><span class="t">{{ e.t }}s</span>
          <span class="lab">{{ e.label || (e.type === 'status' ? '进度' : e.type) }}</span>{{ e.detail || e.text || '' }}</div></div>
      </div>
      <div class="card" v-if="proposal" style="border-color:var(--accent)">
        <h3>估值假设待审批 · {{ proposal.method }}</h3>
        <p class="sub">Agent 建议基准情景每股 <b>{{ proposal.per_share }}</b> 元（现价 {{ proposal.price }} 元）；
          三情景 {{ proposal.scenarios.map((s) => `${s[0]} ${s[1]}`).join(' / ') }}。修改任一假设后会按新假设重新计算并记入证据账本。
          <span v-if="proposal.market_check?.market_implied_ke"><br />市场对照：当前 P/B {{ proposal.market_check.current_pb }}，合理 P/B {{ proposal.market_check.justified_pb }}，
          按现价倒推的股权成本 <b>{{ (proposal.market_check.market_implied_ke * 100).toFixed(1) }}%</b>（模型 Ke {{ (proposal.assumptions.ke * 100).toFixed(2) }}%）</span></p>
        <table><tr><th>假设</th><th>Agent 建议</th><th>审批值</th><th>依据</th></tr>
          <tr v-for="(v, k) in proposal.assumptions" :key="k"><td>{{ k }}</td><td>{{ fmt(v) }}</td>
            <td><input v-model="edits[k]" style="width:200px" /></td><td class="sub" style="text-align:left">{{ proposal.notes[k] || '' }}</td></tr></table>
        <div class="row" style="margin-top:10px">
          <input v-model="reviewer" style="width:120px" placeholder="审批人" />
          <input v-model="comment" style="flex:1" placeholder="审批意见（修改或驳回时必填）" />
          <button @click="decide('approve')">批准</button>
          <button class="ghost" @click="decide('edit')">按修改重算</button>
          <button class="ghost" style="color:var(--red)" @click="decide('reject')">驳回</button>
        </div>
      </div>
      <div class="card memo" v-if="memo">
        <div class="row" style="justify-content:space-between">
          <span class="sub" v-if="stats">用时 {{ stats.seconds }}s · 证据 {{ stats.evidence_items }} 条 · DCF {{ (stats.dcf_runs || []).map((d) => `${d[0]} ${d[1]}`).join(' / ') }}
            · 数字溯源 {{ stats.trace?.numbers_traced }}/{{ stats.trace?.numbers_total }}</span>
          <a v-if="excel" :href="excel"><button class="ghost">下载 Excel 估值模型</button></a>
        </div>
        <div v-html="md(memo)"></div>
      </div>
    </div>
    <div class="card hist">
      <h3>历史研究</h3>
      <div v-for="h in history" :key="h.id" @click="open(h)">
        <b>{{ h.ticker }}</b> <span class="muted">{{ h.date }}</span><br />
        <span class="sub">{{ (h.dcf_runs || []).map((d) => d[1]).join(' / ') }} 元 · 溯源 {{ h.trace }}/{{ h.numbers }}</span>
      </div>
      <div v-if="!history.length" class="muted">暂无</div>
    </div>
  </div>
</template>

<script setup>
import { nextTick, onMounted, ref } from 'vue'
import { api, followJob, md } from '../api'

const ticker = ref('000333'), running = ref(false), events = ref([]), memo = ref(''), stats = ref(null), excel = ref(''), error = ref('')
const history = ref([]), elapsed = ref(0), tl = ref(null)
const review = ref(true), proposal = ref(null), edits = ref({}), reviewer = ref('分析师'), comment = ref(''), jobId = ref('')
const fmt = (v) => (Array.isArray(v) ? v.join(', ') : v)
let timer
onMounted(loadHistory)
async function loadHistory() { history.value = await api('/api/research/history') }
async function start() {
  running.value = true; events.value = []; memo.value = ''; stats.value = null; excel.value = ''; error.value = ''; elapsed.value = 0
  timer = setInterval(() => elapsed.value++, 1000)
  try {
    const { job_id } = await api('/api/research/start', { ticker: ticker.value.trim(), review: review.value })
    jobId.value = job_id
    const res = await followJob(job_id, (e) => {
      if (e.type === 'review') { proposal.value = e.proposal; edits.value = Object.fromEntries(Object.entries(e.proposal.assumptions).map(([k, v]) => [k, fmt(v)])) }
      if (!['done', 'review', 'awaiting_review'].includes(e.type)) { events.value.push(e); nextTick(() => tl.value && (tl.value.scrollTop = 1e6)) }
    })
    memo.value = res.memo; stats.value = res.stats; excel.value = res.files.excel
    loadHistory()
  } catch (e) { error.value = e.message } finally { running.value = false; clearInterval(timer) }
}
async function decide(action) {
  if (action !== 'approve' && !comment.value) { alert('修改或驳回时请填写审批意见'); return }
  const changed = {}
  for (const [k, v] of Object.entries(proposal.value.assumptions)) {
    const raw = String(edits.value[k]).trim()
    if (raw === fmt(v)) continue
    changed[k] = Array.isArray(v) ? raw.split(/[,，\s]+/).filter(Boolean).map(Number) : Number(raw)
  }
  const act = action === 'edit' && !Object.keys(changed).length ? 'approve' : action
  await api(`/api/research/${jobId.value}/review`, { action: act, assumptions: changed, comment: comment.value, reviewer: reviewer.value })
  proposal.value = null
}
async function open(h) {
  memo.value = (await api(`/api/research/memo/${h.id}`)).memo; excel.value = h.excel_url; events.value = []
  stats.value = { seconds: h.seconds, dcf_runs: h.dcf_runs, trace: { numbers_traced: h.trace, numbers_total: h.numbers } }
}
</script>
