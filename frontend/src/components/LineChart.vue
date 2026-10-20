<template>
  <svg :viewBox="`0 0 ${w} ${h}`" style="width:100%;height:auto">
    <g v-for="(s, i) in series" :key="i">
      <polyline :points="pts(s.values)" fill="none" :stroke="s.color" :stroke-width="s.width || 1.6" />
    </g>
    <line :x1="pad" :x2="w - 4" :y1="y(base)" :y2="y(base)" stroke="#ddd" stroke-dasharray="3 3" v-if="base != null" />
    <text v-for="t in ticks" :key="t.v" :x="2" :y="y(t.v) + 3" font-size="9" fill="#888">{{ t.label }}</text>
    <text v-for="(l, i) in xlabels" :key="'x' + i" :x="l.x" :y="h - 2" font-size="9" fill="#888" text-anchor="middle">{{ l.text }}</text>
    <g v-for="(s, i) in series" :key="'lg' + i"><rect :x="pad + i * 150" y="2" width="10" height="3" :fill="s.color" />
      <text :x="pad + 14 + i * 150" y="7" font-size="9" fill="#555">{{ s.name }}</text></g>
  </svg>
</template>
<script setup>
import { computed } from 'vue'
const props = defineProps({ labels: Array, series: Array, base: { type: Number, default: null }, w: { type: Number, default: 760 }, h: { type: Number, default: 220 },
  fmt: { type: Function, default: (v) => v.toFixed(2) } })
const pad = 34
const all = computed(() => props.series.flatMap((s) => s.values).filter((v) => v != null && !isNaN(v)))
const lo = computed(() => Math.min(...all.value)), hi = computed(() => Math.max(...all.value))
const y = (v) => 12 + (props.h - 28) * (1 - (v - lo.value) / ((hi.value - lo.value) || 1))
const x = (i) => pad + (props.w - pad - 6) * (i / Math.max(1, props.labels.length - 1))
const pts = (vals) => vals.map((v, i) => (v == null || isNaN(v) ? null : `${x(i)},${y(v)}`)).filter(Boolean).join(' ')
const ticks = computed(() => [0, 0.5, 1].map((f) => { const v = lo.value + (hi.value - lo.value) * f; return { v, label: props.fmt(v) } }))
const xlabels = computed(() => { const n = props.labels.length, k = 6; return Array.from({ length: k }, (_, j) => { const i = Math.round((n - 1) * j / (k - 1)); return { x: x(i), text: (props.labels[i] || '').slice(0, 7) } }) })
</script>
