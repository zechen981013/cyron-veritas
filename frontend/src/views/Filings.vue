<template>
  <h1>年报问答</h1>
  <div class="sub">巨潮资讯网年报 → 按章节切块（带页码）→ bge-m3 向量 + BM25 → 问题改写与 HyDE → 重排 → 带页码回答 → 引用与数字校验</div>

  <div class="card row">
    <select v-model="key"><option v-for="f in filings" :key="f.key" :value="f.key">{{ f.company }} {{ f.year }} 年报（{{ f.n_chunks }} 块）</option></select>
    <span class="muted">｜入库新年报：</span>
    <input v-model="newTicker" placeholder="代码" style="width:90px" /><input v-model.number="newYear" style="width:80px" />
    <button class="ghost" :disabled="ingesting" @click="ingest"><span v-if="ingesting" class="spin"></span>下载并入库</button>
    <span class="muted">{{ ingestMsg }}</span>
  </div>

  <div class="card">
    <div class="row">
      <input v-model="question" style="flex:1" placeholder="例如：海外收入是多少，占比多少？" @keyup.enter="ask" />
      <button :disabled="asking || !key" @click="ask"><span v-if="asking" class="spin"></span>提问</button>
    </div>
    <div class="row" style="margin-top:8px"><span class="chip" v-for="q in samples" :key="q" @click="question = q; ask()">{{ q }}</span></div>
  </div>

  <div class="grid g2" v-if="answer || evidence.length || status">
    <div class="card">
      <h3>回答 <span class="sub" v-if="status && asking">{{ status }}</span></h3>
      <div class="memo" v-html="rendered" @click="onCite"></div>
      <div v-if="check" class="sub" style="margin-top:8px">
        <span :class="check.ok ? 'ok' : 'bad'">{{ check.ok ? '✓ 引用与数字校验通过' : '✗ 校验未完全通过' }}</span>
        · 引用页 {{ check.cited_pages.join(', ') || '无' }} · 数字溯源 {{ check.numbers_traced }}/{{ check.numbers_total }}
        <span v-if="check.unknown_numbers.length" class="bad"> · 未在原文找到：{{ check.unknown_numbers.join(', ') }}</span>
        <span v-if="check.bad_pages.length" class="bad"> · 引用了证据外页码：{{ check.bad_pages.join(', ') }}</span>
        <span v-if="attempts > 1"> · 重写 {{ attempts - 1 }} 次</span>
      </div>
    </div>
    <div class="card">
      <h3>检索到的证据</h3>
      <div class="sub" v-if="queries.length" style="margin-bottom:6px">检索语句：{{ queries.slice(1).join(' ｜ ').slice(0, 160) }}…</div>
      <div class="ev" v-for="e in evidence" :key="e.id">
        <div class="h"><span class="chip" @click="openPage(e.page_start)">p.{{ e.page_start }}{{ e.page_end !== e.page_start ? '-' + e.page_end : '' }}</span>
          {{ e.section }} {{ e.subsection }} · 重排分 {{ e.score.toFixed(2) }}</div>
        {{ e.text.slice(0, 260) }}{{ e.text.length > 260 ? '…' : '' }}
      </div>
    </div>
  </div>

  <div class="modal" v-if="page" @click.self="page = null">
    <div class="box"><div class="row" style="justify-content:space-between;white-space:normal">
      <b>{{ cur?.company }} {{ cur?.year }} 年报 · 第 {{ page.page }} 页</b>
      <span><a :href="`/api/filings/${cur.ticker}/${cur.year}/pdf#page=${page.page}`" target="_blank">打开 PDF 原文</a>
        <button class="ghost" style="margin-left:8px" @click="page = null">关闭</button></span></div>
      <hr />{{ page.text }}</div>
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { api, followJob, md, postStream } from '../api'

const filings = ref([]), key = ref(''), question = ref(''), asking = ref(false)
const answer = ref(''), evidence = ref([]), queries = ref([]), check = ref(null), status = ref(''), attempts = ref(1), page = ref(null)
const newTicker = ref('600036'), newYear = ref(2025), ingesting = ref(false), ingestMsg = ref('')
const samples = ['海外收入是多少，占营业收入比例多少？', '研发费用和研发人员数量？', '公司如何管理外汇风险？', '2025 年分红和回购情况？', '审计报告的关键审计事项有哪些？']
const cur = computed(() => filings.value.find((f) => f.key === key.value))
const rendered = computed(() => md(answer.value).replace(/\[p\.(\d+)(?:[-–](\d+))?\]/g, (m, a, b) => `<span class="chip" data-p="${a}">p.${a}${b ? '-' + b : ''}</span>`))

onMounted(load)
async function load() { filings.value = await api('/api/filings'); if (!key.value && filings.value.length) key.value = filings.value[0].key }
async function ask() {
  if (!question.value || !cur.value) return
  asking.value = true; answer.value = ''; evidence.value = []; check.value = null; status.value = ''; attempts.value = 1
  try {
    await postStream('/api/qa', { ticker: cur.value.ticker, year: cur.value.year, question: question.value }, (ev) => {
      if (ev.type === 'status') status.value = ev.text
      else if (ev.type === 'evidence') { evidence.value = ev.items; queries.value = ev.queries; status.value = '生成回答' }
      else if (ev.type === 'token') answer.value += ev.text
      else if (ev.type === 'reset') answer.value = ''
      else if (ev.type === 'done') { check.value = ev.check; attempts.value = ev.attempts }
      else if (ev.type === 'error') answer.value = '出错：' + ev.text
    })
  } finally { asking.value = false }
}
function onCite(e) { const p = e.target.dataset?.p; if (p) openPage(+p) }
async function openPage(p) { page.value = await api(`/api/filings/${cur.value.ticker}/${cur.value.year}/page/${p}`) }
async function ingest() {
  ingesting.value = true; ingestMsg.value = ''
  try {
    const { job_id } = await api('/api/filings/ingest', { ticker: newTicker.value, year: newYear.value })
    await followJob(job_id, (e) => e.text && (ingestMsg.value = e.text))
    ingestMsg.value = '入库完成'; await load(); key.value = `${newTicker.value}_${newYear.value}`
  } catch (e) { ingestMsg.value = '失败：' + e.message } finally { ingesting.value = false }
}
</script>
