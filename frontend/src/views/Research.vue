<template>
  <h1>研究备忘录</h1>
  <div class="sub">DeepAgents：研究主管列计划 → 派财务分析师、估值分析师、年报研究员、行业研究员 → 汇总成带证据编号和年报页码的备忘录 + 带公式的 Excel DCF 模型</div>
  <div class="card row">
    <input v-model="ticker" placeholder="A 股代码，如 000333" style="width:180px" @keyup.enter="start" />
    <button :disabled="running" @click="start"><span v-if="running" class="spin"></span>{{ running ? `研究中… ${elapsed}s` : '开始研究' }}</button>
    <span class="muted">首次研究某家公司会先下载并入库最新年报（约 1 分钟），整个流程约 1–2 分钟</span>
    <span class="bad" v-if="error">{{ error }}</span>
  </div>

  <div class="grid g31">
    <div>
      <div class="card" v-if="events.length">
        <h3>Agent 工作过程</h3>
        <div class="timeline" ref="tl"><div v-for="(e, i) in events" :key="i"><span class="t">{{ e.t }}s</span>
          <span class="lab">{{ e.label || (e.type === 'status' ? '进度' : e.type) }}</span>{{ e.detail || e.text || '' }}</div></div>
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
let timer
onMounted(loadHistory)
async function loadHistory() { history.value = await api('/api/research/history') }
async function start() {
  running.value = true; events.value = []; memo.value = ''; stats.value = null; excel.value = ''; error.value = ''; elapsed.value = 0
  timer = setInterval(() => elapsed.value++, 1000)
  try {
    const { job_id } = await api('/api/research/start', { ticker: ticker.value.trim() })
    const res = await followJob(job_id, (e) => { if (e.type !== 'done') { events.value.push(e); nextTick(() => tl.value && (tl.value.scrollTop = 1e6)) } })
    memo.value = res.memo; stats.value = res.stats; excel.value = res.files.excel
    loadHistory()
  } catch (e) { error.value = e.message } finally { running.value = false; clearInterval(timer) }
}
async function open(h) {
  memo.value = (await api(`/api/research/memo/${h.id}`)).memo; excel.value = h.excel_url; events.value = []
  stats.value = { seconds: h.seconds, dcf_runs: h.dcf_runs, trace: { numbers_traced: h.trace, numbers_total: h.numbers } }
}
</script>
