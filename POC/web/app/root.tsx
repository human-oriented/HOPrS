import {
  Link,
  Links,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration,
  json,
  useLoaderData,
  useLocation,
} from "@remix-run/react";
import "./tailwind.css";
import { getToast } from "remix-toast";
import { LinksFunction, LoaderFunctionArgs } from "@remix-run/node";
import { useEffect, useState } from "react";
import { Toaster, toast as notify } from "sonner";
import * as Icon from 'react-feather';

export const loader = async ({ request }: LoaderFunctionArgs) => {
  const { toast, headers } = await getToast(request);
  return json({ toast }, { headers });
}

export function Layout({ children }: { children: React.ReactNode }) {
  const data = useLoaderData<typeof loader>();
  const [menuActive, setMenuActive] = useState(false)
  const {pathname} = useLocation()

  useEffect(() => {
    if (data?.toast?.type === "error") {
      notify.error(data.toast.message);
    }
    if (data?.toast?.type === "success") {
      notify.success(data.toast.message);
    }
  }, [data?.toast]);

  useEffect(() => {
    setMenuActive(false)
  },[pathname])

  const d = new Date()

  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <Meta />
        <Links />
      </head>
      <body className="overflow-x-hidden">
          <header className="flex flex-row items-center justify-between w-full shadow-md px-5 fixed top-0 bg-white z-20 py-3 lg:py-0">
              <Link to="/"><img src="/hoprs-logo.png" className="w-[220px] lg:w-[400px] " /></Link>
              <div className="flex flex-row items-center">
                  <div className="flex flex-row items-center mx-3 font-bold border-r border-black pr-3">
                      <span className="text-xs mr-1 hidden lg:inline">Test User</span>
                      <div className="w-6 h-6 lg:w-8 lg:h-8 rounded-full flex items-center justify-center bg-gray-700 text-white">
                          <Icon.User size={18} />
                      </div>
                  </div>
                  <button className="z-20" onClick={() => setMenuActive(!menuActive)}>
                    <Icon.Menu size={30} />
                  </button>
              </div>
          </header>
          <div className={`${menuActive ? '-translate-x-2/3 lg:-translate-x-1/4' : 'translate-x-0'} transition-transform	`}>
            <div className={`translate-x-full w-2/3 lg:w-1/4 bg-white h-full min-h-screen absolute pt-[100px] right-0 z-10 text-black transition-transform`}>
              <div className="h-screen sticky top-0 right-0 -mt-[100px] p-5 pt-[100px]">
              <ul className="text-xl font-bold flex flex-col justify-between h-full">
                <span>
                  <li className={`${pathname == '/' ? 'border-b border-gray-200 text-gray-500 cursor-default' : ''} py-3`}><Link className="" to="/">HOME</Link></li>
                  <li className={`${pathname == '/encode' ? 'border-b border-gray-200 text-gray-500 cursor-default' : ''} py-3`}><Link className="" to="/encode">ENCODE</Link></li>
                  <li className={`${pathname == '/demo-compare' ? 'border-b border-gray-200 text-gray-500 cursor-default' : ''} py-3`}><Link className="" to="/demo-compare">COMPARE</Link></li>
                </span>
                <span>
                  <li className="my-3"><a className="flex flex-row items-center w-full" href="https://www.human-oriented.org/" target="_blank"><span>LEARN MORE</span> <Icon.ExternalLink className="ml-1" /></a></li>
                  <p className="text-xs font-normal">&copy;{d.getFullYear()} by Human Oriented Proof Standard</p>
                </span>
                
              </ul>
              </div>
            </div>
            <div>
              
              {children}
              <ScrollRestoration />
              <Scripts />
              <Toaster />
              <div className={`${menuActive ? 'flex' : 'hidden'} w-full h-full absolute top-0 left-0 bg-black opacity-10 z-10 cursor-pointer`} onClick={() => setMenuActive(!menuActive)}></div>
            </div>
          </div>
      </body>
    </html>
  );
}

export default function App() {
  return <Outlet />;
}
