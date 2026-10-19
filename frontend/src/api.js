import { marked } from 'marked'
import DOMPurify from 'dompurify'

export async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
  })
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText)
  return r.json()
}

// 订阅后台任务的进度事件（SSE）；结束后取最终结果
export function followJob(jobId, onEvent) {
  return new Promise((resolve, reject) => {
    const es = new EventSource(`/api/jobs/${jobId}/events`)
    es.onmessage = (m) => {
      const ev = JSON.parse(m.data)
      onEvent?.(ev)
      if (ev.type === 'done' || ev.type === 'error') {
        es.close()
        api(`/api/jobs/${jobId}`).then((j) => (j.status === 'done' ? resolve(j.result) : reject(new Error(j.error))))
      }
    }
    es.onerror = () => { es.close(); api(`/api/jobs/${jobId}`).then((j) => j.status === 'done' ? resolve(j.result) : j.status === 'error' ? reject(new Error(j.error)) : followJob(jobId, onEvent).then(resolve, reject)) }
  })
}

// POST + SSE（年报问答）：fetch 读流，按 data: 行解析
export async function postStream(path, body, onEvent) {
  const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
  const reader = r.body.getReader()
  const dec = new TextDecoder()
  let buf = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buf += dec.decode(value, { stream: true })
    let i
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i).trim()
      buf = buf.slice(i + 1)
      if (line.startsWith('data:')) onEvent(JSON.parse(line.slice(5)))
    }
  }
}

export const md = (s) => DOMPurify.sanitize(marked.parse(s || ''))
export const pct = (x, d = 1) => (x == null ? '—' : `${(x * 100).toFixed(d)}%`)
export const wan = (x) => (x == null ? '—' : `${(x / 1e4).toLocaleString('zh-CN', { maximumFractionDigits: 1, minimumFractionDigits: 1 })} 万元`)
