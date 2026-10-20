<template>
  <h1>资管中台</h1>
  <div class="sub">托管估值表 → 净值与估值核对 → 每日风控与合同限额 → 业绩归因 → 股票池与投委会 → 交易前检查 → 投资者月报</div>
  <div class="card row">
    <select v-model="code" @change="loadAll"><option v-for="p in products" :key="p.code" :value="p.code">{{ p.name }}</option></select>
    <span v-for="t in tabs" :key="t" class="chip" :style="tab === t ? 'background:var(--accent);color:#fff' : ''" @click="tab = t">{{ t }}</span>
  </div>

  <template v-if="ov">
    <!-- 产品概览 -->
    <template v-if="tab === '产品概览'">
      <div class="grid g5" style="margin-top:14px">
        <div class="kpi"><span>单位净值</span><b>{{ m.unit_nav.toFixed(4) }}</b><span>{{ m.end }}</span></div>
        <div class="kpi"><span>成立以来 / 年化</span><b>{{ pct(m.total_return) }}</b><span>年化 {{ pct(m.ann_return) }}</span></div>
        <div class="kpi"><span>年化超额（沪深300）</span><b :class="m.ann_excess >= 0 ? 'ok' : 'bad'">{{ pct(m.ann_excess) }}</b><span>信息比率 {{ m.info_ratio.toFixed(2) }}</span></div>
        <div class="kpi"><span>最大回撤</span><b>{{ pct(m.max_drawdown) }}</b><span>夏普 {{ m.sharpe.toFixed(2) }} · 卡玛 {{ m.calmar.toFixed(2) }}</span></div>
        <div class="kpi"><span>规模 / 待处理风控事件</span><b>{{ (ov.nav.at(-1).net_assets / 1e8).toFixed(2) }} 亿</b><span :class="ov.open_events ? 'bad' : 'ok'">{{ ov.open_events }} 条待处理</span></div>
      </div>
      <div class="card"><h3>单位净值 vs 沪深300（归一）</h3>
        <LineChart :labels="ov.nav.map((r) => r.date)" :base="1" :fmt="(v) => v.toFixed(2)"
          :series="[{ name: '单位净值', values: ov.nav.map((r) => r.unit_nav), color: '#0E7C66' }, { name: '沪深300', values: ov.nav.map((r) => r.bench), color: '#aaa', width: 1.2 }]" /></div>
      <div class="grid g2">
        <div class="card"><h3>合同要素</h3>
          <table><tr><td>管理人 / 托管人</td><td>{{ ov.product.manager }} / {{ ov.product.custodian }}</td></tr>
            <tr><td>成立日 / 基准</td><td>{{ ov.product.inception }} / 沪深300</td></tr>
            <tr><td>管理费 / 托管费 / 业绩报酬</td><td>{{ pct(ov.product.fees.management) }} / {{ pct(ov.product.fees.custody, 2) }} / {{ pct(ov.product.fees.performance, 0) }}</td></tr>
            <tr><td>预警线 / 止损线</td><td>{{ ov.product.contract_limits.warning_line }} / {{ ov.product.contract_limits.stop_line }}</td></tr>
            <tr><td>单票 / 单行业 / 现金</td><td>≤ {{ pct(ov.product.contract_limits.single_name_max, 0) }} / ≤ {{ pct(ov.product.contract_limits.industry_max, 0) }} / ≥ {{ pct(ov.product.contract_limits.cash_min, 0) }}</td></tr></table></div>
        <div class="card"><h3>申购赎回（登记机构）</h3>
          <table><tr><th>日期</th><th>类型</th><th>金额</th><th>确认净值</th><th>份额</th></tr>
            <tr v-for="f in ov.flows" :key="f.date"><td>{{ f.date }}</td><td>{{ f.type === 'subscribe' ? '申购' : '赎回' }}</td><td>{{ wan(f.amount) }}</td><td>{{ f.nav.toFixed(4) }}</td><td>{{ (f.shares / 1e4).toFixed(1) }} 万份</td></tr></table></div>
      </div>
      <div class="card"><h3>月度收益</h3>
        <table><tr><th>月份</th><th>产品</th><th>沪深300</th><th>超额</th></tr>
          <tr v-for="r in ov.monthly.slice(-12).reverse()" :key="r.month"><td>{{ r.month }}</td><td :class="r['组合'] < 0 ? 'neg' : ''">{{ pct(r['组合']) }}</td><td>{{ pct(r['基准']) }}</td><td :class="r['超额'] < 0 ? 'neg' : 'ok'">{{ pct(r['超额']) }}</td></tr></table></div>
    </template>

    <!-- 风控合规 -->
    <template v-if="tab === '风控合规'">
      <div class="grid g5" style="margin-top:14px">
        <div class="kpi"><span>1 日 95% VaR</span><b>{{ pct(r.var95_1d) }}</b><span>CVaR {{ pct(r.cvar95_1d) }}</span></div>
        <div class="kpi"><span>年化波动 / Beta</span><b>{{ pct(r.vol_ann, 1) }}</b><span>Beta {{ r.beta?.toFixed(2) }}</span></div>
        <div class="kpi"><span>股票仓位 / 现金</span><b>{{ pct(r.stock_weight, 1) }}</b><span>现金 {{ pct(r.cash_weight, 1) }}</span></div>
        <div class="kpi"><span>流动性（5 日可变现）</span><b>{{ pct(r.liquidity?.['5'] ?? r.liquidity?.[5], 0) }}</b><span>1 日 {{ pct(r.liquidity?.['1'] ?? r.liquidity?.[1], 0) }}</span></div>
        <div class="kpi"><span>距预警线 / 止损线</span><b>{{ pct(r.to_warning_line, 1) }}</b><span>止损线 {{ pct(r.to_stop_line, 1) }}</span></div>
      </div>
      <div class="card"><h3>合同限额（{{ r.date }}）</h3>
        <table><tr><th>限额</th><th>实际</th><th>限值</th><th>说明</th><th>状态</th></tr>
          <tr v-for="l in r.limits" :key="l.rule"><td>{{ l.label }}</td><td>{{ l.rule.includes('line') ? l.value.toFixed(4) : (l.rule === 'banned' ? l.value : pct(l.value, 1)) }}</td>
            <td>{{ l.direction === 'min' ? '≥' : '≤' }} {{ l.rule.includes('line') || l.rule === 'banned' ? l.limit : pct(l.limit, 0) }}</td><td>{{ l.detail }}</td><td><span :class="['tagx', l.status]">{{ l.status }}</span></td></tr></table></div>
      <div class="grid g2">
        <div class="card"><h3>行业分布（申万一级，占净值）</h3>
          <div v-for="(v, k) in r.by_industry" :key="k" class="row" style="gap:6px;margin:2px 0"><span style="width:80px">{{ k }}</span>
            <div :style="{ width: v * 600 + 'px', height: '9px', background: v > ov.product.contract_limits.industry_max * 0.9 ? 'var(--amber)' : 'var(--accent)', borderRadius: '3px' }"></div><span class="muted">{{ pct(v, 1) }}</span></div></div>
        <div class="card"><h3>待处理风控事件（{{ events.length }}）</h3>
          <div class="timeline" style="max-height:300px"><div v-for="e in events.slice(0, 40)" :key="e.id">
            <span class="t" style="width:80px">{{ e.date }}</span><span :class="['tagx', e.status]">{{ e.status }}</span> {{ e.detail }}
            <span class="chip" @click="handle(e)">确认处理</span></div></div></div>
      </div>
    </template>

    <!-- 业绩归因 -->
    <template v-if="tab === '业绩归因' && att">
      <div class="card"><h3>Brinson 归因（累计，各月加总）</h3>
        <table><tr><th v-for="k in attCols" :key="k">{{ k }}</th></tr><tr><td v-for="k in attCols" :key="k">{{ pct(att.brinson_total[k]) }}</td></tr></table>
        <p class="sub">基准总收益 = 沪深300 实际涨跌；行业收益 = 申万一级行业指数；基准行业权重 = 沪深300 最新成分权重（近似），由此产生的差单列「基准近似误差」。</p></div>
      <div class="card"><h3>逐月</h3>
        <table><tr><th>月份</th><th v-for="k in attCols" :key="k">{{ k }}</th><th>超配最多</th><th>低配最多</th></tr>
          <tr v-for="b in att.brinson.slice().reverse()" :key="b.month"><td>{{ b.month }}</td><td v-for="k in attCols" :key="k" :class="b[k] < 0 ? 'neg' : ''">{{ pct(b[k]) }}</td><td>{{ b['超配最多'] }}</td><td>{{ b['低配最多'] }}</td></tr></table></div>
      <div class="card"><h3>风格暴露（中证 800 截面 z 分数）</h3>
        <table><tr><th>因子</th><th>组合</th><th>沪深300</th><th>主动暴露</th></tr>
          <tr v-for="s in att.style" :key="s.factor"><td>{{ s.factor }}</td><td>{{ s['组合'].toFixed(2) }}</td><td>{{ s['基准（沪深300）'].toFixed(2) }}</td><td :class="Math.abs(s['主动暴露']) > 0.5 ? 'bad' : ''">{{ s['主动暴露'].toFixed(2) }}</td></tr></table></div>
    </template>

    <!-- 股票池与投委会 -->
    <template v-if="tab === '股票池与投委会' && pool">
      <div class="grid g2">
        <div class="card"><h3>核心池（{{ pool.pools.filter((p) => p.pool === '核心池').length }}）</h3>
          <table><tr><th>名称</th><th>代码</th><th>入池日</th><th>依据</th></tr>
            <tr v-for="p in pool.pools.filter((p) => p.pool === '核心池')" :key="p.code"><td>{{ p.name }}</td><td>{{ p.code }}</td><td>{{ p.since }}</td><td class="sub" style="text-align:left">{{ p.reason }}</td></tr></table>
          <h3 style="margin-top:12px">禁投池</h3><div v-for="b in pool.banned" :key="b.code" class="sub">{{ b.code }}：{{ b.reason }}</div></div>
        <div class="card"><h3>提交投委会</h3>
          <div class="row"><input v-model="icForm.code" placeholder="代码" style="width:90px" /><input v-model="icForm.name" placeholder="名称" style="width:100px" />
            <select v-model="icForm.request"><option>入核心池</option><option>调出核心池</option><option>入禁投池</option></select></div>
          <div class="row" style="margin-top:6px"><input v-model="icForm.material" placeholder="申请材料 / 研究备忘录链接" style="flex:1" /></div>
          <div class="row" style="margin-top:6px"><select v-model="icForm.decision"><option>批准</option><option>否决</option><option>补充材料</option></select>
            <input v-model="icForm.note" placeholder="决议说明" style="flex:1" /><button @click="submitIC">记录决议</button></div>
          <h3 style="margin-top:12px">最近决议</h3>
          <div class="timeline"><div v-for="d in pool.ic.slice(0, 30)" :key="d.id"><span class="t" style="width:80px">{{ d.decided_at }}</span>
            <span class="lab">{{ d.decision }}</span>{{ d.name }}（{{ d.code }}）{{ d.request }} · {{ d.requested_by }}</div></div></div>
      </div>
    </template>

    <!-- 交易前检查 -->
    <template v-if="tab === '交易前检查'">
      <div class="card"><h3>拟下单指令（每行：代码 买/卖 数量）</h3>
        <textarea v-model="orderText" rows="5" style="width:100%;font-family:monospace;border:1px solid var(--line);border-radius:6px;padding:8px"></textarea>
        <div class="row" style="margin-top:6px"><button @click="checkOrders">检查</button><span class="sub">按最新估值日收盘价模拟成交，检查禁投池、核心池、预警线、单票、行业、现金</span></div></div>
      <div class="card" v-if="pre"><h3>结果：<span :class="pre.pass ? 'ok' : 'bad'">{{ pre.pass ? '全部通过' : '不通过' }}</span></h3>
        <table><tr><th>代码</th><th>方向</th><th>数量</th><th>价格</th><th>结果</th><th>原因</th></tr>
          <tr v-for="o in pre.orders" :key="o.code + o.side"><td>{{ o.code }}</td><td>{{ o.side === 'buy' ? '买' : '卖' }}</td><td>{{ o.qty }}</td><td>{{ o.price?.toFixed(2) }}</td>
            <td :class="o.pass ? 'ok' : 'bad'">{{ o.pass ? '通过' : '拒绝' }}</td><td style="text-align:left">{{ o.reasons.join('；') }}</td></tr></table>
        <div v-for="c in pre.portfolio_checks" :key="c" class="bad" style="margin-top:4px">组合层面：{{ c }}</div></div>
    </template>

    <!-- 估值表与对账 -->
    <template v-if="tab === '估值表与对账'">
      <div class="grid g2">
        <div class="card"><h3>估值核对（托管净值 vs 自估净值）</h3>
          <table><tr><th>日期</th><th>差异</th><th>解释项</th><th>状态</th></tr>
            <tr v-for="x in ov.reconciliation" :key="x.date"><td>{{ x.date }}</td><td>{{ x.diff.toFixed(2) }}</td>
              <td style="text-align:left" class="sub">{{ Object.entries(typeof x.explained === 'string' ? JSON.parse(x.explained) : x.explained).map(([k, v]) => `${k} ${v}`).join('；') }}</td>
              <td><span :class="['tagx', x.status === '一致' ? '正常' : '超限']">{{ x.status }}</span></td></tr></table></div>
        <div class="card"><h3>托管估值表（最近 30 个交易日）</h3>
          <div v-for="s in sheets" :key="s.name"><a :href="s.url">{{ s.name }}</a></div></div>
      </div>
      <div class="card"><h3>投资者月报</h3>
        <div class="row"><input v-model="month" style="width:100px" /><button :disabled="reporting" @click="makeReport"><span v-if="reporting" class="spin"></span>生成月报</button>
          <a v-if="reportUrl" :href="reportUrl" target="_blank">打开 PDF</a></div></div>
    </template>
  </template>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { api, followJob, pct, wan } from '../api'
import LineChart from '../components/LineChart.vue'

const tabs = ['产品概览', '风控合规', '业绩归因', '股票池与投委会', '交易前检查', '估值表与对账']
const products = ref([]), code = ref('CV-VG1'), tab = ref('产品概览')
const ov = ref(null), events = ref([]), att = ref(null), pool = ref(null), sheets = ref([]), pre = ref(null)
const m = computed(() => ov.value?.metrics), r = computed(() => ov.value?.risk || {})
const attCols = ['组合收益', '基准收益', '超额', '配置效应', '选股效应', '交互效应', '其他（交易与现金）', '基准近似误差']
const orderText = ref('000002 买 10000\n600519 买 100\n600030 买 1000')
const icForm = ref({ code: '', name: '', request: '入核心池', material: '', decision: '批准', note: '' })
const month = ref('2026-09'), reporting = ref(false), reportUrl = ref('')

onMounted(async () => { products.value = await api('/api/fund/products'); loadAll() })
async function loadAll() {
  ov.value = await api(`/api/fund/${code.value}/overview`)
  events.value = await api(`/api/fund/${code.value}/events`)
  pool.value = await api(`/api/fund/${code.value}/pool`)
  sheets.value = await api(`/api/fund/${code.value}/sheets`)
  api(`/api/fund/${code.value}/attribution`).then((x) => (att.value = x))
}
async function handle(e) {
  const note = prompt(`处理说明（${e.detail}）`, '已知悉，下次调仓降低行业集中度'); if (!note) return
  await api(`/api/fund/${code.value}/events/${e.id}/handle`, { handled_by: '风控岗', note }); events.value = await api(`/api/fund/${code.value}/events`)
}
async function checkOrders() {
  const orders = orderText.value.trim().split('\n').map((l) => l.trim().split(/\s+/)).filter((x) => x.length >= 3)
    .map(([c, s, q]) => ({ code: c, side: s.startsWith('买') || s === 'buy' ? 'buy' : 'sell', qty: Number(q) }))
  pre.value = await api(`/api/fund/${code.value}/pretrade`, { orders })
}
async function submitIC() { await api(`/api/fund/${code.value}/ic`, { ...icForm.value, votes: { 赞成: 3, 反对: 0, 弃权: 0 } }); pool.value = await api(`/api/fund/${code.value}/pool`) }
async function makeReport() {
  reporting.value = true; reportUrl.value = ''
  try { const { job_id } = await api(`/api/fund/${code.value}/report`, { month: month.value }); const res = await followJob(job_id); reportUrl.value = res.pdf || res.html }
  finally { reporting.value = false }
}
</script>
