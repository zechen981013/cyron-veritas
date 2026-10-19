<template>
  <h1>业绩会纪要</h1>
  <div class="sub">巨潮「投资者关系活动记录表」或业绩会录音（本机 Whisper 转写）→ 指引、经营数据、投资者关注点、回避的问题 → 与上一次对比；每条都带原文引用并校验</div>

  <div class="card row">
    <input v-model="ticker" style="width:120px" placeholder="000333" @keyup.enter="loadRecords" />
    <button class="ghost" @click="loadRecords">查找交流记录</button>
    <select v-if="records.length" v-model="recordId" style="max-width:360px">
      <option v-for="r in records" :key="r.record_id" :value="r.record_id">{{ r.date }} · {{ r.title }}</option>
    </select>
    <label v-if="records.length" class="muted"><input type="checkbox" v-model="compare" /> 与上一次对比</label>
    <button v-if="records.length" :disabled="running" @click="run"><span v-if="running" class="spin"></span>生成纪要</button>
    <span class="bad" v-if="error">{{ error }}</span>
  </div>

  <div class="card row">
    <span class="muted">或上传业绩会录音：</span>
    <input type="file" accept="audio/*,.m4a,.mp3,.wav,.aiff" @change="(e) => (audio = e.target.files[0])" />
    <input v-model="company" style="width:140px" placeholder="公司名（可选）" />
    <button class="ghost" :disabled="running || !audio" @click="runAudio">转写并生成纪要</button>
  </div>

  <div class="card" v-if="events.length">
    <div class="timeline"><div v-for="(e, i) in events" :key="i"><span class="t">{{ e.t }}s</span>{{ e.text }}</div></div>
  </div>
  <div class="card" v-if="check">
    <span :class="check.ok ? 'ok' : 'bad'">{{ check.ok ? '✓ 引用与数字校验通过' : '✗ 有未通过校验的条目（已在纪要中标注）' }}</span>
    <span class="sub"> · 原文引用 {{ check.quotes_ok }}/{{ check.items }} · 数字 {{ check.numbers_traced }}/{{ check.numbers_total }}
      <template v-if="cmpCheck"> · 对比引用 {{ cmpCheck.quotes_ok }}/{{ cmpCheck.changes }}</template></span>
  </div>
  <div class="card memo" v-if="markdown" v-html="md(markdown)"></div>
  <div class="card" v-if="transcript"><h3>转写原文</h3><div class="sub" style="white-space:pre-wrap">{{ transcript }}</div></div>
</template>

<script setup>
import { ref } from 'vue'
import { api, followJob, md } from '../api'

const ticker = ref('000333'), records = ref([]), recordId = ref(''), compare = ref(true), running = ref(false), error = ref('')
const events = ref([]), markdown = ref(''), check = ref(null), cmpCheck = ref(null), transcript = ref(''), audio = ref(null), company = ref('')

async function loadRecords() {
  error.value = ''
  try { records.value = await api(`/api/calls/${ticker.value.trim()}/records`); recordId.value = records.value[0]?.record_id || '' }
  catch (e) { error.value = e.message }
}
async function start(promise) {
  running.value = true; events.value = []; markdown.value = ''; check.value = null; cmpCheck.value = null; transcript.value = ''; error.value = ''
  try {
    const { job_id } = await promise
    const res = await followJob(job_id, (e) => e.text && events.value.push(e))
    markdown.value = res.markdown; check.value = res.check; cmpCheck.value = res.compare?.check; transcript.value = res.transcript || ''
  } catch (e) { error.value = e.message } finally { running.value = false }
}
const run = () => start(api('/api/calls/run', { ticker: ticker.value.trim(), record_id: recordId.value, compare: compare.value }))
function runAudio() {
  const fd = new FormData(); fd.append('file', audio.value); fd.append('company', company.value)
  start(fetch('/api/calls/audio', { method: 'POST', body: fd }).then((r) => r.json()))
}
</script>
