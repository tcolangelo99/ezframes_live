// Preact, MIT License
var n,l,u,i,t,o,r={},f=[],e=/acit|ex(?:s|g|n|p|$)|rph|grid|ows|mnc|ntw|ine[ch]|zoo|^ord|itera/i;function c(e,n){for(var t in n)e[t]=n[t];return e}function s(e){var n=e.parentNode;n&&n.removeChild(e)}function a(e,n,t){var _,l,o,r=arguments,i={};for(o in n)"key"==o?_=n[o]:"ref"==o?l=n[o]:i[o]=n[o];if(arguments.length>3)for(t=[t],o=3;o<arguments.length;o++)t.push(r[o]);if(null!=t&&(i.children=t),"function"==typeof e&&null!=e.defaultProps)for(o in e.defaultProps)void 0===i[o]&&(i[o]=e.defaultProps[o]);return v(e,i,_,l,null)}function v(e,t,_,l,o){var r={type:e,props:t,key:_,ref:l,__k:null,__:null,__b:0,__e:null,__d:void 0,__c:null,__h:null,constructor:void 0,__v:null==o?++n.__v:o};return null!=n.vnode&&n.vnode(r),r}function h(){return{current:null}}function y(e){return e.children}function p(e,n){this.props=e,this.context=n}function d(e,n){if(null==n)return e.__?d(e.__,e.__.__k.indexOf(e)+1):null;for(var t;n<e.__k.length;n++)if(null!=(t=e.__k[n])&&null!=t.__e)return t.__e;return"function"==typeof e.type?d(e):null}function _(e){var n,t;if(null!=(e=e.__)&&null!=e.__c){for(e.__e=e.__c.base=null,n=0;n<e.__k.length;n++)if(null!=(t=e.__k[n])&&null!=t.__e){e.__e=e.__c.base=t.__e;break}return _(e)}}function k(e){(!e.__d&&(e.__d=!0)&&u.push(e)&&!b.__r++||t!==n.debounceRendering)&&((t=n.debounceRendering)||i)(b)}function b(){for(var e;b.__r=u.length;)e=u.sort(function(e,n){return e.__v.__b-n.__v.__b}),u=[],e.some(function(e){var n,t,l,o,r,i;e.__d&&(r=(o=(n=e).__v).__e,(i=n.__P)&&(t=[],(l=c({},o)).__v=o.__v+1,I(i,o,l,n.__n,void 0!==i.ownerSVGElement,null!=o.__h?[r]:null,t,null==r?d(o):r,o.__h),T(t,o),o.__e!=r&&_(o)))})}function m(e,n,t,_,l,o,i,u,s,c){var p,a,h,m,k,b,C,P=_&&_.__k||f,S=P.length;for(t.__k=[],p=0;p<n.length;p++)if(null!=(m=t.__k[p]=null==(m=n[p])||"boolean"==typeof m?null:"string"==typeof m||"number"==typeof m||"bigint"==typeof m?v(null,m,null,null,m):Array.isArray(m)?v(y,{children:m},null,null,null):m.__b>0?v(m.type,m.props,m.key,null,m.__v):m)){if(m.__=t,m.__b=t.__b+1,null===(h=P[p])||h&&m.key==h.key&&m.type===h.type)P[p]=void 0;else for(a=0;a<S;a++){if((h=P[a])&&m.key==h.key&&m.type===h.type){P[a]=void 0;break}h=null}I(e,m,h=h||r,l,o,i,u,s,c),k=m.__e,(a=m.ref)&&h.ref!=a&&(C||(C=[]),h.ref&&C.push(h.ref,null,m),C.push(a,m.__c||k,m)),null!=k?(null==b&&(b=k),"function"==typeof m.type&&null!=m.__k&&m.__k===h.__k?m.__d=s=g(m,s,e):s=x(e,m,h,P,k,s),c||"option"!==t.type?"function"==typeof t.type&&(t.__d=s):e.value=""):s&&h.__e==s&&s.parentNode!=e&&(s=d(h))}for(t.__e=b,p=S;p--;)null!=P[p]&&("function"==typeof t.type&&null!=P[p].__e&&P[p].__e==t.__d&&(t.__d=d(_,p+1)),L(P[p],P[p]));if(C)for(p=0;p<C.length;p++)z(C[p],C[++p],C[++p])}function g(e,n,t){var _,l;for(_=0;_<e.__k.length;_++)(l=e.__k[_])&&(l.__=e,n="function"==typeof l.type?g(l,n,t):x(t,l,l,e.__k,l.__e,n));return n}function w(e,n){return n=n||[],null==e||"boolean"==typeof e||(Array.isArray(e)?e.some(function(e){w(e,n)}):n.push(e)),n}function x(e,n,t,_,l,o){var r,i,u;if(void 0!==n.__d)r=n.__d,n.__d=void 0;else if(null==t||l!=o||null==l.parentNode)e:if(null==o||o.parentNode!==e)e.appendChild(l),r=null;else{for(i=o,u=0;(i=i.nextSibling)&&u<_.length;u+=2)if(i==l)break e;e.insertBefore(l,o),r=o}return void 0!==r?r:l.nextSibling}function A(e,n,t,_,l){var o;for(o in t)"children"===o||"key"===o||o in n||C(e,o,null,t[o],_);for(o in n)l&&"function"!=typeof n[o]||"children"===o||"key"===o||"value"===o||"checked"===o||t[o]===n[o]||C(e,o,n[o],t[o],_)}function P(n,t,_){"-"===t[0]?n.setProperty(t,_):n[t]=null==_?"":"number"!=typeof _||e.test(t)?_:_+"px"}function C(e,n,t,_,l){var o;e:if("style"===n)if("string"==typeof t)e.style.cssText=t;else{if("string"==typeof _&&(e.style.cssText=_=""),_)for(n in _)t&&n in t||P(e.style,n,"");if(t)for(n in t)_&&t[n]===_[n]||P(e.style,n,t[n])}else if("o"===n[0]&&"n"===n[1])o=n!==(n=n.replace(/Capture$/,"")),n=n.toLowerCase()in e?n.toLowerCase().slice(2):n.slice(2),e.l||(e.l={}),e.l[n+o]=t,t?_||e.addEventListener(n,o?H:$,o):e.removeEventListener(n,o?H:$,o);else if("dangerouslySetInnerHTML"!==n){if(l)n=n.replace(/xlink[H:h]/,"h").replace(/sName$/,"s");else if("href"!==n&&"list"!==n&&"form"!==n&&"tabIndex"!==n&&"download"!==n&&n in e)try{e[n]=null==t?"":t;break e}catch(e){}"function"==typeof t||(null!=t&&(!1!==t||"a"===n[0]&&"r"===n[1])?e.setAttribute(n,t):e.removeAttribute(n))}}function $(e){this.l[e.type+!1](n.event?n.event(e):e)}function H(e){this.l[e.type+!0](n.event?n.event(e):e)}function I(e,t,_,l,o,r,i,u,s){var f,a,d,h,v,k,g,b,C,x,P,S=t.type;if(void 0!==t.constructor)return null;null!=_.__h&&(s=_.__h,u=t.__e=_.__e,t.__h=null,r=[u]),(f=n.__b)&&f(t);try{e:if("function"==typeof S){if(b=t.props,C=(f=S.contextType)&&l[f.__c],x=f?C?C.props.value:f.__:l,_.__c?g=(a=t.__c=_.__c).__=a.__E:("prototype"in S&&S.prototype.render?t.__c=a=new S(b,x):(t.__c=a=new p(b,x),a.constructor=S,a.render=M),C&&C.sub(a),a.props=b,a.state||(a.state={}),a.context=x,a.__n=l,d=a.__d=!0,a.__h=[]),null==a.__s&&(a.__s=a.state),null!=S.getDerivedStateFromProps&&(a.__s==a.state&&(a.__s=c({},a.__s)),c(a.__s,S.getDerivedStateFromProps(b,a.__s))),h=a.props,v=a.state,d)null==S.getDerivedStateFromProps&&null!=a.componentWillMount&&a.componentWillMount(),null!=a.componentDidMount&&a.__h.push(a.componentDidMount);else{if(null==S.getDerivedStateFromProps&&b!==h&&null!=a.componentWillReceiveProps&&a.componentWillReceiveProps(b,x),!a.__e&&null!=a.shouldComponentUpdate&&!1===a.shouldComponentUpdate(b,a.__s,x)||t.__v===_.__v){a.props=b,a.state=a.__s,t.__v!==_.__v&&(a.__d=!1),a.__v=t,t.__e=_.__e,t.__k=_.__k,t.__k.forEach(function(e){e&&(e.__=t)}),a.__h.length&&i.push(a);break e}null!=a.componentWillUpdate&&a.componentWillUpdate(b,a.__s,x),null!=a.componentDidUpdate&&a.__h.push(function(){a.componentDidUpdate(h,v,k)})}a.context=x,a.props=b,a.state=a.__s,(f=n.__r)&&f(t),a.__d=!1,a.__v=t,a.__P=e,f=a.render(a.props,a.state,a.context),a.state=a.__s,null!=a.getChildContext&&(l=c(c({},l),a.getChildContext())),d||null==a.getSnapshotBeforeUpdate||(k=a.getSnapshotBeforeUpdate(h,v)),P=null!=f&&f.type===y&&null==f.key?f.props.children:f,m(e,Array.isArray(P)?P:[P],t,_,l,o,r,i,u,s),a.base=t.__e,t.__h=null,a.__h.length&&i.push(a),g&&(a.__E=a.__=null),a.__e=!1}else null==r&&t.__v===_.__v?(t.__k=_.__k,t.__e=_.__e):t.__e=j(_.__e,t,_,l,o,r,i,s);(f=n.diffed)&&f(t)}catch(e){t.__v=null,(s||null!=r)&&(t.__e=u,t.__h=!!s,r[r.indexOf(u)]=null),n.__e(e,t,_)}}function T(e,t){n.__c&&n.__c(t,e),e.some(function(t){try{e=t.__h,t.__h=[],e.some(function(e){e.call(t)})}catch(e){n.__e(e,t.__v)}})}function j(e,n,t,_,l,o,i,u){var c,p,a,d,h=t.props,v=n.props,y=n.type,k=0;if("svg"===y&&(l=!0),null!=o)for(;k<o.length;k++)if((c=o[k])&&(c===e||(y?c.localName==y:3==c.nodeType))){e=c,o[k]=null;break}if(null==e){if(null===y)return document.createTextNode(v);e=l?document.createElementNS("http://www.w3.org/2000/svg",y):document.createElement(y,v.is&&v),o=null,u=!1}if(null===y)h===v||u&&e.data===v||(e.data=v);else{if(o=o&&f.slice.call(e.childNodes),p=(h=t.props||r).dangerouslySetInnerHTML,a=v.dangerouslySetInnerHTML,!u){if(null!=o)for(h={},d=0;d<e.attributes.length;d++)h[e.attributes[d].name]=e.attributes[d].value;(a||p)&&(a&&(p&&a.__html==p.__html||a.__html===e.innerHTML)||(e.innerHTML=a&&a.__html||""))}if(A(e,v,h,l,u),a)n.__k=[];else if(k=n.props.children,m(e,Array.isArray(k)?k:[k],n,t,_,l&&"foreignObject"!==y,o,i,e.firstChild,u),null!=o)for(k=o.length;k--;)null!=o[k]&&s(o[k]);u||("value"in v&&void 0!==(k=v.value)&&(k!==e.value||"progress"===y&&!k)&&C(e,"value",k,h.value,!1),"checked"in v&&void 0!==(k=v.checked)&&k!==e.checked&&C(e,"checked",k,h.checked,!1))}return e}function z(e,t,_){try{"function"==typeof e?e(t):e.current=t}catch(e){n.__e(e,_)}}function L(e,t,_){var l,o,r;if(n.unmount&&n.unmount(e),(l=e.ref)&&(l.current&&l.current!==e.__e||z(l,null,t)),_||"function"==typeof e.type||(_=null!=(o=e.__e)),e.__e=e.__d=void 0,null!=(l=e.__c)){if(l.componentWillUnmount)try{l.componentWillUnmount()}catch(e){n.__e(e,t)}l.base=l.__P=null}if(l=e.__k)for(r=0;r<l.length;r++)l[r]&&L(l[r],t,_);null!=o&&s(o)}function M(e,n,t){return this.constructor(e,t)}function N(e,t,_){var l,o,i;n.__&&n.__(e,t),o=(l="function"==typeof _)?null:_&&_.__k||t.__k,i=[],I(t,e=(!l&&_||t).__k=a(y,null,[e]),o||r,r,void 0!==t.ownerSVGElement,!l&&_?[_]:o?null:t.firstChild?f.slice.call(t.childNodes):null,i,!l&&_?_:o?o.__e:t.firstChild,l),T(i,e)}function O(e,n){N(e,n,O)}function S(e,n,t){var _,l,o,r=arguments,i=c({},e.props);for(o in n)"key"==o?_=n[o]:"ref"==o?l=n[o]:i[o]=n[o];if(arguments.length>3)for(t=[t],o=3;o<arguments.length;o++)t.push(r[o]);return null!=t&&(i.children=t),v(e.type,i,_||e.key,l||e.ref,null)}function q(e,n){var t={__c:n="__cC"+o++,__:e,Consumer:function(e,n){return e.children(n)},Provider:function(e){var t,_;return this.getChildContext||(t=[],(_={})[n]=this,this.getChildContext=function(){return _},this.shouldComponentUpdate=function(e){this.props.value!==e.value&&t.some(k)},this.sub=function(e){t.push(e);var n=e.componentWillUnmount;e.componentWillUnmount=function(){t.splice(t.indexOf(e),1),n&&n.call(e)}}),e.children}};return t.Provider.__=t.Consumer.contextType=t}n={__e:function(e,n){for(var t,_,l;n=n.__;)if((t=n.__c)&&!t.__)try{if((_=t.constructor)&&null!=_.getDerivedStateFromError&&(t.setState(_.getDerivedStateFromError(e)),l=t.__d),null!=t.componentDidCatch&&(t.componentDidCatch(e),l=t.__d),l)return t.__E=t}catch(n){e=n}throw e},__v:0},l=function(e){return null!=e&&void 0===e.constructor},p.prototype.setState=function(e,n){var t;t=null!=this.__s&&this.__s!==this.state?this.__s:this.__s=c({},this.state),"function"==typeof e&&(e=e(c({},t),this.props)),e&&c(t,e),null!=e&&this.__v&&(n&&this.__h.push(n),k(this))},p.prototype.forceUpdate=function(e){this.__v&&(this.__e=!0,e&&this.__h.push(e),k(this))},p.prototype.render=y,u=[],i="function"==typeof Promise?Promise.prototype.then.bind(Promise.resolve()):setTimeout,b.__r=0,o=0;export{N as render,O as hydrate,a as createElement,a as h,y as Fragment,h as createRef,l as isValidElement,p as Component,S as cloneElement,q as createContext,w as toChildArray,n as options};
