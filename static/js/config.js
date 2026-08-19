/* Backend location - empty for same-origin, absolute for split hosting (DEPLOY.md) */

const BACKEND_URL = "";

const apiUrl = path => BACKEND_URL.replace(/\/$/, "") + path;

export { BACKEND_URL, apiUrl };
