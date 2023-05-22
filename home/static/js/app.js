
let toast = document.querySelector('.toast')
toast.timeOut = setTimeout(() => toast.remove(), 10000)
let close = toast.querySelector('i')
close.onclick = function(){
    toast.remove()
}

