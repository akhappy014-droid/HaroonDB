const form=document.getElementById('chatForm');
const input=document.getElementById('messageInput');
const messages=document.getElementById('messages');
const welcome=document.getElementById('welcome');
const sendBtn=document.getElementById('sendBtn');
const statusEl=document.getElementById('modelStatus');
const sidebar=document.getElementById('sidebar');
const overlay=document.getElementById('overlay');
let historyItems=[];

function addMessage(role, content, extraClass=''){
  welcome.style.display='none';
  const row=document.createElement('div');
  row.className=`message ${role} ${extraClass}`;
  const avatar=document.createElement('div');
  avatar.className='avatar';
  avatar.textContent=role==='user'?'أنت':'H';
  const body=document.createElement('div');
  body.className='message-content';
  body.textContent=content;
  row.append(avatar,body); messages.appendChild(row);
  document.getElementById('chatArea').scrollTop=document.getElementById('chatArea').scrollHeight;
  return row;
}
async function sendMessage(text){
  text=text.trim(); if(!text || sendBtn.disabled) return;
  addMessage('user',text); historyItems.push({role:'user',content:text});
  input.value=''; input.style.height='auto'; sendBtn.disabled=true;
  const typing=addMessage('assistant','يفكر Haroon...','typing');
  try{
    const res=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:text,history:historyItems.slice(-8,-1)})});
    const data=await res.json();
    typing.remove();
    if(!res.ok) throw new Error(data.error||'حدث خطأ أثناء الاتصال بالنموذج.');
    const answer=data.answer||'لم يرجع النموذج إجابة.';
    addMessage('assistant',answer); historyItems.push({role:'assistant',content:answer});
  }catch(err){
    typing.remove();
    addMessage('assistant',err.message);
  }finally{sendBtn.disabled=false;input.focus();}
}
form.addEventListener('submit',e=>{e.preventDefault();sendMessage(input.value)});
input.addEventListener('input',()=>{input.style.height='auto';input.style.height=Math.min(input.scrollHeight,180)+'px'});
document.querySelectorAll('[data-prompt]').forEach(b=>b.addEventListener('click',()=>{input.value=b.dataset.prompt;sendMessage(input.value)}));
document.getElementById('newChat').addEventListener('click',()=>{
  messages.innerHTML='';historyItems=[];welcome.style.display='block';sidebar.classList.remove('open');overlay.classList.remove('show');
});
document.getElementById('menuBtn').addEventListener('click',()=>{sidebar.classList.add('open');overlay.classList.add('show')});
overlay.addEventListener('click',()=>{sidebar.classList.remove('open');overlay.classList.remove('show')});
fetch('/api/status').then(r=>r.json()).then(d=>{
  statusEl.textContent=d.ready?'النموذج المدرّب جاهز':'الواجهة جاهزة — الأوزان تحتاج تدريبًا';
  statusEl.previousElementSibling.style.background=d.ready?'#5cc79b':'#e0a64b';
}).catch(()=>statusEl.textContent='تعذر فحص حالة النموذج');
