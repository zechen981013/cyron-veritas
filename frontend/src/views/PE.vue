<template>
  <div class="card row" style="margin-top:0">
    <span v-for="t in tabs" :key="t" class="chip" :style="tab === t ? 'background:var(--accent);color:#fff' : ''" @click="tab = t">{{ t }}</span>
  </div>
  <div v-if="!ov" class="card"><span class="spin"></span> 加载中…</div>
  <template v-else>
    <!-- 基金概览 -->
    <template v-if="tab === '基金概览'">
      <div class="grid g5" style="margin-top:14px">
        <div class="kpi"><span>认缴 / 实缴</span><b>{{ yi(ov.product.commitment) }}</b><span>LP 实缴 {{ pct(L.called_pct, 0) }}（{{ yi(L.lp_paid_in) }}）</span></div>
        <div class="kpi"><span>TVPI（LP 净）</span><b>{{ L.TVPI.toFixed(2) }}x</b><span>项目毛回报 {{ L.gross_moic.toFixed(2) }}x</span></div>
        <div class="kpi"><span>DPI / RVPI</span><b>{{ L.DPI.toFixed(2) }}x</b><span>RVPI {{ L.RVPI.toFixed(2) }}x</span></div>
        <div class="kpi"><span>净 IRR / 毛 IRR</span><b>{{ pct(L.net_irr, 1) }}</b><span>毛 {{ pct(L.gross_irr, 1) }}</span></div>
        <div class="kpi"><span>KS-PME</span><b>{{ L.pme['沪深300']?.toFixed(2) }}</b><span>沪深300 · 中证500 {{ L.pme['中证500']?.toFixed(2) }}</span></div>
      </div>
      <div class="card" style="border-left:3px solid var(--red)"><h3>先看这个：后见之明压力测试</h3>
        <p class="sub" style="margin:0 0 8px">组合里 6 个真实项目都是后来上市成功的科创板公司——投资的时候并不知道。假设每个真实项目只有一定概率按实际路径上市，
          否则按未上市项目的典型结局（40% 核销、40% 回购只收回约七成、20% 并购或转让 1.2x；比例是假设），看组合毛回报倍数的分布（1 万次模拟）。</p>
        <table><tr><th>上市成功率</th><th>平均</th><th>P10</th><th>中位数</th><th>P90</th><th>亏损概率</th></tr>
          <tr v-for="s in ov.stress" :key="s.ipo_rate"><td>{{ pct(s.ipo_rate, 0) }}</td><td>{{ s.mean.toFixed(2) }}x</td><td>{{ s.p10.toFixed(2) }}x</td>
            <td>{{ s.p50.toFixed(2) }}x</td><td>{{ s.p90.toFixed(2) }}x</td><td :class="s.prob_below_1 > 0.2 ? 'bad' : ''">{{ pct(s.prob_below_1, 0) }}</td></tr></table></div>
      <div class="card"><h3>TVPI / DPI 与 J 曲线（季度，LP 净）</h3>
        <LineChart :labels="ov.series.map((r) => r.date)" :base="1" :fmt="(v) => v.toFixed(1) + 'x'"
          :series="[{ name: 'TVPI', values: ov.series.map((r) => r.TVPI), color: '#0E7C66' }, { name: 'DPI', values: ov.series.map((r) => r.DPI), color: '#d4a017' }]" />
        <LineChart :labels="ov.series.map((r) => r.date)" :base="0" :h="160" :fmt="(v) => (v / 1e8).toFixed(1) + '亿'"
          :series="[{ name: 'LP 累计净现金流（J 曲线）', values: ov.series.map((r) => r.cum_net_cf), color: '#1b2430' }]" /></div>
      <div class="card"><h3>投资组合</h3>
        <table><tr><th>项目</th><th>行业</th><th>交割</th><th>上市</th><th>退出完成</th><th>投资（万）</th><th>回收（万）</th><th>剩余估值（万）</th><th>MOIC</th><th>IRR</th><th>状态</th></tr>
          <tr v-for="d in ov.deals" :key="d.deal" @click="openDeal(d.deal)" style="cursor:pointer">
            <td style="text-align:left">{{ d.company }} <span class="muted">{{ d.synthetic ? '虚构' : d.deal }}</span></td><td>{{ d.sector }}</td><td>{{ d.close }}</td>
            <td>{{ d.listing || '—' }}</td><td>{{ d.exit_end || '—' }}</td><td>{{ wan0(d.invested) }}</td><td>{{ wan0(d.realized) }}</td><td>{{ wan0(d.fv) }}</td>
            <td :class="d.moic < 1 ? 'neg' : 'ok'">{{ d.moic?.toFixed(2) }}x</td><td>{{ d.irr == null ? '—' : pct(d.irr, 0) }}</td><td>{{ d.status }}</td></tr></table>
        <p class="sub">点击项目看交易条款（招股书出处）、季度估值和交易流水。费用合计：管理费 {{ wan0(-(ov.fees['管理费'] || 0)) }} 万，基金费用 {{ wan0(-(ov.fees['基金费用'] || 0)) }} 万。</p></div>
      <div class="card"><h3>基金条款</h3>
        <table><tr><td>首次交割 / 投资期 / 退出期</td><td>{{ ov.product.inception }} / {{ ov.product.terms.investment_period_years }} 年 / {{ ov.product.terms.exit_period_years }} 年（可延 {{ ov.product.terms.extension_years }} 年）</td></tr>
          <tr><td>管理费</td><td>投资期认缴额 × {{ pct(ov.product.terms.mgmt_fee_invest, 0) }}，退出期未退出成本 × {{ pct(ov.product.terms.mgmt_fee_exit, 0) }}，半年预收</td></tr>
          <tr><td>分配</td><td>欧式整体分配：返还实缴 → {{ pct(ov.product.terms.hurdle, 0) }} 门槛收益（复利）→ GP 追补 {{ pct(ov.product.terms.catch_up, 0) }} → {{ 100 - ov.product.terms.carry * 100 }}/{{ ov.product.terms.carry * 100 }}</td></tr>
          <tr><td>减持纪律</td><td>解禁后 {{ ov.product.exit_policy.sell_months }} 个月内分批；日成交量 ≤ {{ pct(ov.product.exit_policy.max_volume_share, 0) }}；90 天 ≤ 总股本 {{ pct(ov.product.exit_policy.max_total_share_90d, 0) }}；低于成本暂缓、投委会复议</td></tr></table></div>
    </template>

    <!-- 项目管线 -->
    <template v-if="tab === '项目管线' && pipe">
      <div class="grid g5" style="margin-top:14px">
        <div class="kpi" v-for="s in funnel" :key="s.k"><span>{{ s.k }}</span><b>{{ s.n }}</b><span>{{ s.note }}</span></div>
      </div>
      <div class="card"><h3>项目管线（线索 → 立项 → 尽调 → 投决 → 交割 → 投后 → 退出）</h3>
        <table><tr><th>公司</th><th>阶段</th><th>状态</th><th style="text-align:left">过程</th><th style="text-align:left">未投原因 / 事后回看</th></tr>
          <tr v-for="p in pipe" :key="p.deal"><td style="text-align:left">{{ p.company }} <span class="muted">{{ p.synthetic ? '虚构' : p.deal }}</span>
              <a v-if="!p.synthetic" class="chip" @click="ddCode = p.deal; tab = '尽调报告'">尽调</a></td>
            <td>{{ p.stage }}</td><td>{{ p.status }}</td>
            <td style="text-align:left;font-size:11px">{{ p.events.filter((e) => !e.event.startsWith('投委会复议')).map((e) => e.event + ' ' + e.date.slice(0, 7)).join(' → ') }}
              <span v-if="p.events.some((e) => e.event.startsWith('投委会复议'))" class="muted">（投委会复议 {{ p.events.filter((e) => e.event.startsWith('投委会复议')).length }} 次：股价低于成本暂缓减持）</span></td>
            <td style="text-align:left;font-size:11px">{{ p.reason }}<br v-if="p.reason" /><span class="muted">{{ p.events.find((e) => e.event.includes('回看'))?.note || '' }}</span></td></tr></table>
        <p class="sub">真实公司的「线索、立项、投决」时间和未投原因是模拟的；上市日期、发行价、最新股价是真实数据。</p></div>
    </template>

    <!-- 投资组合与估值 -->
    <template v-if="tab === '投资组合与估值'">
      <div class="card row" style="margin-top:14px"><span>项目：</span>
        <select v-model="dealId" @change="openDeal(dealId)"><option v-for="d in ov.deals" :key="d.deal" :value="d.deal">{{ d.company }}</option></select></div>
      <template v-if="deal">
        <div class="card" v-if="deal.terms"><h3>交易条款（来自招股说明书，代码计算）</h3>
          <table><tr><td>跟投轮次</td><td>{{ deal.terms.round.date }} {{ deal.terms.round.kind }}，{{ deal.terms.round.price }} {{ deal.terms.round.unit }}；{{ deal.terms.round.investors }}
              <a :href="`/prospectus/${deal.terms.code}.pdf#page=${deal.terms.pages.round}`" target="_blank" class="chip">第 {{ deal.terms.pages.round }} 页</a></td></tr>
            <tr><td>投后估值 / 投资额 / 持股</td><td>{{ (deal.terms.post_money / 1e8).toFixed(2) }} 亿
              <span v-if="deal.terms.cross_check?.stated && deal.terms.cross_check?.price_x_capital" :class="deal.terms.cross_check.confirmed ? 'ok' : 'bad'" style="font-size:11px">
                （原文估值与「价格 × 注册资本 {{ (deal.terms.cross_check.price_x_capital / 1e8).toFixed(2) }} 亿」{{ deal.terms.cross_check.confirmed ? '一致' : '不一致，待复核' }}）</span>
              <span v-else-if="!deal.terms.cross_check?.stated" class="muted" style="font-size:11px">（原文未写估值，按价格 × 本轮后注册资本计算）</span>
              / {{ wan0(deal.terms.amount) }} 万 / {{ pct(deal.terms.ownership_at_close, 3) }}
              <span v-if="deal.terms.dilution.length" class="muted">（后续增资稀释到 {{ pct(deal.terms.ownership_pre_ipo, 3) }}）</span></td></tr>
            <tr><td>发行前总股本 → 持股数 → 每股成本</td><td>{{ (deal.terms.pre_ipo_shares / 1e4).toFixed(2) }} 万股 → {{ (deal.terms.shares / 1e4).toFixed(1) }} 万股 → {{ deal.terms.cost_per_share.toFixed(2) }} 元
              <a :href="`/prospectus/${deal.terms.code}.pdf#page=${deal.terms.pages.pre_ipo_shares}`" target="_blank" class="chip">第 {{ deal.terms.pages.pre_ipo_shares }} 页</a></td></tr>
            <tr><td>上市 / 发行价</td><td>{{ deal.terms.listing_date }} / {{ deal.terms.issue_price }} 元（{{ (deal.terms.issue_price / deal.terms.cost_per_share).toFixed(1) }} 倍于每股成本）</td></tr>
            <tr><td>解禁</td><td>{{ deal.terms.unlock_date }}：{{ deal.terms.lockup_note }} <span v-if="deal.terms.lockup_review" class="tagx 预警">待复核</span></td></tr></table></div>
        <div class="card"><h3>季度估值</h3>
          <LineChart :labels="deal.marks.map((m) => m.date)" :base="null" :fmt="(v) => (v / 1e4).toFixed(0) + '万'"
            :series="[{ name: '公允价值', values: deal.marks.map((m) => m.fv), color: '#0E7C66' }, { name: '剩余成本', values: deal.marks.map((m) => m.cost), color: '#aaa', width: 1.2 }]" />
          <div class="timeline" style="max-height:260px"><table><tr><th>季末</th><th>状态</th><th style="text-align:left">方法</th><th>单价</th><th>流动性折扣</th><th>公允价值（万）</th></tr>
            <tr v-for="m in deal.marks.slice().reverse()" :key="m.date"><td>{{ m.date }}</td><td>{{ m.status }}</td><td style="text-align:left;font-size:11px">{{ m.method }} <span class="muted">{{ m.note }}</span></td>
              <td>{{ m.price ? m.price.toFixed(2) : '—' }}</td><td>{{ m.discount ? pct(m.discount, 1) : '—' }}</td><td>{{ wan0(m.fv) }}</td></tr></table></div></div>
        <div class="card"><h3>交易流水</h3><table><tr><th>日期</th><th>类型</th><th>数量</th><th>金额（万）</th><th style="text-align:left">说明</th></tr>
          <tr v-for="(t, i) in deal.tx" :key="i"><td>{{ t.date }}</td><td>{{ t.type }}</td><td>{{ t.qty ? (t.qty / 1e4).toFixed(1) + ' 万股' : '' }}</td><td :class="t.amount < 0 ? 'neg' : ''">{{ wan0(t.amount) }}</td>
            <td style="text-align:left;font-size:11px">{{ t.note }}</td></tr></table></div>
      </template>
    </template>

    <!-- 出资与分配 -->
    <template v-if="tab === '出资与分配' && cap">
      <div class="grid g2" style="margin-top:14px">
        <div class="card"><h3>分配瀑布（累计，LP 类别）</h3>
          <div v-for="(v, k) in cap.waterfall" :key="k" class="row" style="gap:6px;margin:4px 0"><span style="width:140px">{{ k }}</span>
            <div :style="{ width: v / maxTier * 300 + 'px', height: '10px', background: k.includes('GP') ? '#d4a017' : 'var(--accent)', borderRadius: '3px' }"></div><span class="muted">{{ yi(v) }}</span></div>
          <p class="sub">GP 已收业绩报酬 {{ yi((cap.waterfall['GP 追补'] || 0) + (cap.waterfall['超额收益（GP 20%）'] || 0)) }}，应计未付 {{ yi(cap.accrued_carry) }}。
            追补把 GP 补到全部已分配利润的 20%。</p></div>
        <div class="card"><h3>合伙人资本账户</h3>
          <table><tr><th>合伙人</th><th>认缴</th><th>实缴</th><th>分配</th><th>净值</th><th>TVPI</th></tr>
            <tr v-for="a in cap.accounts" :key="a.id"><td style="text-align:left">{{ a.name }} <span class="muted">{{ a.type }}</span></td><td>{{ wan0(a.commitment) }}</td><td>{{ wan0(a.called) }}</td>
              <td>{{ wan0(a.distributed) }}</td><td>{{ a.nav == null ? '—' : wan0(a.nav) }}</td><td>{{ a.tvpi?.toFixed(2) }}x</td></tr></table>
          <p class="sub">金额单位：万元。GP 跟投部分不交管理费、不参与业绩报酬，所以倍数比 LP 高。</p></div>
      </div>
      <div class="card"><h3>缴款与分配记录</h3>
        <div class="timeline" style="max-height:420px"><table><tr><th>日期</th><th>通知日</th><th>类型</th><th>金额（万）</th><th style="text-align:left">用途 / 分配档</th></tr>
          <tr v-for="(e, i) in cap.events.slice().reverse()" :key="i"><td>{{ e.date }}</td><td>{{ e.notice_date }}</td><td>{{ e.type }}</td><td :class="e.type === '缴款' ? 'neg' : 'ok'">{{ wan0(e.amount) }}</td>
            <td style="text-align:left;font-size:11px">{{ e.purpose || Object.entries(e.tiers).filter(([, v]) => Math.abs(v) >= 1).map(([k, v]) => k + ' ' + wan0(v) + ' 万').join('；') }}</td></tr></table></div></div>
    </template>

    <!-- 尽调报告 -->
    <template v-if="tab === '尽调报告'">
      <div class="card row" style="margin-top:14px"><span>公司：</span>
        <select v-model="ddCode" @change="loadDD"><option v-for="p in (pipe || []).filter((x) => !x.synthetic)" :key="p.deal" :value="p.deal">{{ p.company }}（{{ p.status }}）</option></select>
        <a v-if="dd" :href="`/prospectus/${dd.code}.pdf`" target="_blank">打开招股说明书</a></div>
      <template v-if="dd">
        <div class="card"><h3>{{ dd.company }}：{{ dd.items.business.summary }}</h3>
          <p class="sub">行业：{{ dd.items.business.industry }} · 上市 {{ dd.listing_date }} · 尽调 Agent 按清单抽取 7 项，每个数字核对引用页；
            <span :class="dd.unverified.length ? 'bad' : 'ok'">{{ dd.unverified.length ? dd.unverified.length + ' 个字段待分析师复核' : '全部核对通过' }}</span></p>
          <div v-if="dd.unverified.length" class="muted" style="font-size:12px">待复核：{{ dd.unverified.map((u) => `${u.field}=${u.value}（第 ${u.page} 页）`).join('；') }}</div></div>
        <div class="card"><h3>股本演变与融资历史</h3>
          <table><tr><th>时间</th><th>事项</th><th>价格</th><th>注册资本 / 股本（万）</th><th>估值（亿）</th><th style="text-align:left">投资方</th><th>出处</th></tr>
            <tr v-for="(r, i) in dd.items.rounds.rounds" :key="i" :class="isBad('rounds[' + i + ']') ? 'bad' : ''"><td>{{ r.date }}</td><td>{{ r.kind }}</td>
              <td>{{ r.price != null ? r.price + ' ' + (r.unit || '') : '—' }}</td><td>{{ r.capital_before ?? '' }}{{ r.capital_after != null ? ' → ' + r.capital_after : '' }}</td>
              <td>{{ r.valuation != null ? r.valuation + '（' + (r.valuation_type || '') + '）' : '—' }}</td>
              <td style="text-align:left;font-size:11px">{{ r.investors }}<br /><span class="muted">{{ r.note }}</span></td>
              <td><a :href="`/prospectus/${dd.code}.pdf#page=${r.page}`" target="_blank" class="chip">p{{ r.page }}</a></td></tr></table></div>
        <div class="grid g2">
          <div class="card"><h3>发行条款与财务摘要</h3>
            <table><tr><td>发行前总股本</td><td>{{ (dd.items.ipo.pre_ipo_shares.value / 1e4).toFixed(2) }} 万股 <a :href="`/prospectus/${dd.code}.pdf#page=${dd.items.ipo.pre_ipo_shares.page}`" target="_blank" class="chip">p{{ dd.items.ipo.pre_ipo_shares.page }}</a></td></tr>
              <tr><td>发行价 / 发行股数</td><td>{{ dd.items.ipo.issue_price.value }} 元 / {{ dd.items.ipo.issue_shares.value }} 万股</td></tr></table>
            <table style="margin-top:8px"><tr><th>年度</th><th>营业收入（万）</th><th>归母净利润</th><th>扣非净利润</th><th>出处</th></tr>
              <tr v-for="y in dd.items.financials.years" :key="y.year"><td>{{ y.year }}</td><td>{{ fmt(y.revenue) }}</td><td :class="y.net_profit < 0 ? 'neg' : ''">{{ fmt(y.net_profit) }}</td>
                <td :class="y.net_profit_deducted < 0 ? 'neg' : ''">{{ fmt(y.net_profit_deducted) }}</td><td><a :href="`/prospectus/${dd.code}.pdf#page=${y.page}`" target="_blank" class="chip">p{{ y.page }}</a></td></tr></table>
            <p class="sub">前五大客户占营业收入 {{ dd.items.customers.top5_share }}%（{{ dd.items.customers.year }} 年，第 {{ dd.items.customers.page }} 页）</p></div>
          <div class="card"><h3>主要风险</h3>
            <div v-for="r in dd.items.risks.risks" :key="r.title" style="margin:6px 0"><b>{{ r.title }}</b> <a :href="`/prospectus/${dd.code}.pdf#page=${r.page}`" target="_blank" class="chip">p{{ r.page }}</a>
              <div class="muted" style="font-size:12px">{{ r.summary }}</div></div>
            <h3 style="margin-top:12px">股东锁定承诺（按股东）</h3>
            <div v-for="(v, k) in dd.items.lockup_by_investor || {}" :key="k" style="font-size:12px;margin:4px 0"><b>{{ k }}</b>：
              {{ v.found ? [v.months_from_listing && `上市后 ${v.months_from_listing} 个月`, v.months_from_acquisition && `取得后 ${v.months_from_acquisition} 个月`].filter(Boolean).join(' / ') + `（第 ${v.page} 页）` : '招股书里没找到，待复核' }}
              <div class="muted" v-if="v.quote">「{{ v.quote }}」</div></div></div>
        </div>
      </template>
    </template>

    <!-- LP 季报 -->
    <template v-if="tab === 'LP 季报'">
      <div class="card" style="margin-top:14px"><h3>LP 季度报告</h3>
        <div class="row"><select v-model="rq.lp"><option v-for="l in ov.product.lps.filter((x) => x.type !== 'GP')" :key="l.id" :value="l.id">{{ l.name }}</option></select>
          <input v-model="rq.quarter" style="width:90px" /><button :disabled="reporting" @click="makeReport"><span v-if="reporting" class="spin"></span>生成季报</button>
          <a v-if="reportUrl" :href="reportUrl" target="_blank">打开 PDF</a></div>
        <p class="sub">资本账户、投资组合估值、本季度出资分配与退出、GP 观点（大模型只写观点，数字全部由代码填入占位符，并校验方向词与「分配不改变 TVPI」这类口径）。</p>
        <div v-if="reportCheck" class="muted" style="font-size:12px">校验：占位符 {{ reportCheck.placeholders }} 个，{{ reportCheck.ok ? '全部通过' : '未通过：' + JSON.stringify(reportCheck) }}</div></div>
    </template>
  </template>
</template>

<script setup>
import { computed, onMounted, ref, watch } from 'vue'
import { api, followJob, pct } from '../api'
import LineChart from '../components/LineChart.vue'

const props = defineProps({ code: String, initialTab: String })
const tabs = ['基金概览', '项目管线', '投资组合与估值', '出资与分配', '尽调报告', 'LP 季报']
const tab = ref(tabs.includes(props.initialTab) ? props.initialTab : '基金概览')
const ov = ref(null), pipe = ref(null), cap = ref(null), deal = ref(null), dealId = ref(''), dd = ref(null), ddCode = ref('688234')
const rq = ref({ lp: 'LP02', quarter: '2023Q3' }), reporting = ref(false), reportUrl = ref(''), reportCheck = ref(null)
const L = computed(() => ov.value?.latest)
const yi = (v) => (v / 1e8).toFixed(2) + ' 亿'
const wan0 = (v) => (v == null ? '—' : (v / 1e4).toLocaleString('zh-CN', { maximumFractionDigits: 0 }))
const fmt = (v) => (v == null ? '—' : v.toLocaleString('zh-CN', { maximumFractionDigits: 0 }))
const maxTier = computed(() => Math.max(...Object.values(cap.value?.waterfall || { a: 1 })))
const isBad = (prefix) => (dd.value?.unverified || []).some((u) => u.field.startsWith(prefix))
const funnel = computed(() => {
  const ps = pipe.value || []
  const reached = (s) => ps.filter((p) => p.events.some((e) => e.event.startsWith(s))).length
  return [{ k: '线索', n: reached('线索'), note: '进入管线的公司' }, { k: '尽调', n: reached('尽调'), note: '招股书 / 资料尽调' },
    { k: '投决', n: reached('投决'), note: `否决 ${ps.filter((p) => p.events.some((e) => e.event === '投决：否决')).length} 个` },
    { k: '交割', n: reached('交割'), note: '含 2 个虚构未上市项目' }, { k: '已退出', n: ps.filter((p) => p.stage === '退出').length, note: '减持完毕 / 回购 / 核销' }]
})

async function load() {
  ov.value = await api(`/api/pe/${props.code}/overview`)
  api(`/api/pe/${props.code}/pipeline`).then((x) => (pipe.value = x))
  api(`/api/pe/${props.code}/capital`).then((x) => (cap.value = x))
  loadDD()
  if (ov.value.deals.length) openDeal(ov.value.deals[0].deal, false)
}
async function openDeal(id, jump = true) { dealId.value = id; deal.value = await api(`/api/pe/${props.code}/deal/${id}`); if (jump) tab.value = '投资组合与估值' }
async function loadDD() { dd.value = null; dd.value = await api(`/api/pe/dd/${ddCode.value}`) }
async function makeReport() {
  reporting.value = true; reportUrl.value = ''; reportCheck.value = null
  try { const { job_id } = await api(`/api/pe/${props.code}/report`, rq.value); const r = await followJob(job_id); reportUrl.value = r.pdf || r.html; reportCheck.value = r.check }
  finally { reporting.value = false }
}
watch(ddCode, () => { if (tab.value === '尽调报告') loadDD() })
onMounted(load)
</script>
