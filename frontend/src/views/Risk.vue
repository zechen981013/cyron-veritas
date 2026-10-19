<template>
  <h1>组合风控日报</h1>
  <div class="sub">读取多券商持仓 → 代码计算 VaR、集中度、压力测试和限额 → 大模型写解读 → 数字溯源校验</div>
  <div class="card row">
    <select v-model="portfolio"><option v-for="p in portfolios" :key="p.name" :value="p.name">{{ p.name }}（{{ p.positions }} 个持仓，{{ p.accounts.length }} 个账户）</option></select>
    <button :disabled="running" @click="run"><span v-if="running" class="spin"></span>{{ running ? '生成中…' : '生成今日日报' }}</button>
    <span class="muted" v-if="steps.length">{{ steps[steps.length - 1].text }}</span>
    <span class="bad" v-if="error">{{ error }}</span>
  </div>

  <template v-if="r">
    <div class="grid g5" style="margin-top:14px">
      <div class="kpi"><span>组合总市值</span><b>{{ wan(f.total_value) }}</b><span>{{ f.as_of }} · {{ f.n_positions }} 个持仓</span></div>
      <div class="kpi"><span>1 日 95% VaR</span><b>{{ pct(f.var95_hist, 2) }}</b><span>约 {{ wan(f.var95_amount) }}</span></div>
      <div class="kpi"><span>年化波动率</span><b>{{ pct(f.vol_ann) }}</b><span>Beta(沪深300) {{ f.beta_csi300.toFixed(2) }}</span></div>
      <div class="kpi"><span>当前回撤</span><b>{{ pct(f.current_drawdown) }}</b><span>最大 {{ pct(f.max_drawdown) }}</span></div>
      <div class="kpi"><span>限额</span><b :class="breaches ? 'bad' : 'ok'">{{ breaches }} 项超限</b><span>{{ warns }} 项预警</span></div>
    </div>

    <div class="grid g31">
      <div class="card">
        <h3>解读</h3>
        <div class="ai">
          <p style="margin-top:0">{{ r.narrative.summary }}</p>
          <b>主要风险</b><ul><li v-for="x in r.narrative.key_risks" :key="x">{{ x }}</li></ul>
          <template v-if="r.narrative.limit_comments.length"><b>超限与预警</b><ul><li v-for="x in r.narrative.limit_comments" :key="x">{{ x }}</li></ul></template>
          <b>需要复核</b><ul><li v-for="x in r.narrative.review_items" :key="x">{{ x }}</li></ul>
          <div class="sub">来源：{{ r.source }} · 数字溯源 {{ r.check.numbers_traced }}/{{ r.check.numbers_total }}
            <span :class="r.check.ok ? 'ok' : 'bad'">{{ r.check.ok ? '✓ 全部可在计算结果中找到' : '✗ 有未溯源数字' }}</span></div>
        </div>
      </div>
      <div class="card">
        <h3>流程</h3>
        <div class="timeline"><div v-for="(s, i) in steps" :key="i"><span class="t">{{ s.t }}s</span>{{ s.text }}</div></div>
      </div>
    </div>

    <div class="card">
      <h3>限额检查</h3>
      <table><tr><th>限额</th><th>限值</th><th>实际</th><th>说明</th><th>状态</th></tr>
        <tr v-for="l in r.limits" :key="l.key"><td>{{ l.label }}</td><td>{{ l.direction === 'min' ? '≥' : '≤' }} {{ pct(l.limit) }}</td><td>{{ pct(l.value) }}</td><td>{{ l.note }}</td><td><span :class="['tagx', l.status]">{{ l.status }}</span></td></tr>
      </table>
    </div>

    <div class="grid g2">
      <div class="card"><h3>风险贡献（占组合波动）</h3>
        <div v-for="(v, k) in f.risk_contribution" :key="k" class="row" style="gap:6px;margin:3px 0">
          <span style="width:80px">{{ k }}</span><div :style="{ width: v * 300 + 'px', height: '10px', background: v > 0.3 ? 'var(--red)' : 'var(--accent)', borderRadius: '3px' }"></div><span class="muted">{{ pct(v) }}</span></div>
      </div>
      <div class="card"><h3>压力测试</h3>
        <table><tr><th>情景</th><th>组合损益</th><th>金额</th></tr>
          <tr v-for="(v, k) in f.stress" :key="k"><td>{{ k }} <span class="muted">{{ v.type }}</span></td><td class="neg">{{ pct(v.pnl_pct) }}</td><td>{{ wan(v.pnl_amount) }}</td></tr></table>
        <p class="sub">VaR 回测：{{ f.var_backtest.days }} 天突破 {{ f.var_backtest.exceptions }} 次（理论 {{ f.var_backtest.expected.toFixed(1) }} 次），Kupiec p = {{ f.var_backtest.p_value.toFixed(2) }}</p>
      </div>
    </div>
    <div class="card"><h3>完整报告（含图表）<a :href="r.report_url" target="_blank" style="font-weight:400;font-size:12px;margin-left:8px">新窗口打开</a></h3><iframe :src="r.report_url"></iframe></div>
  </template>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { api, followJob, pct, wan } from '../api'

const portfolios = ref([]), portfolio = ref('sample_portfolio.csv')
const running = ref(false), steps = ref([]), r = ref(null), error = ref('')
const f = computed(() => r.value?.facts)
const breaches = computed(() => r.value?.limits.filter((l) => l.status === '超限').length)
const warns = computed(() => r.value?.limits.filter((l) => l.status === '预警').length)

onMounted(async () => { portfolios.value = await api('/api/portfolios') })
async function run() {
  running.value = true; steps.value = []; error.value = ''
  try {
    const { job_id } = await api('/api/risk/run', { portfolio: portfolio.value })
    r.value = await followJob(job_id, (e) => e.type === 'step' && steps.value.push(e))
  } catch (e) { error.value = e.message } finally { running.value = false }
}
</script>
