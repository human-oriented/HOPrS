import {
  Links,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration,
  json,
  useLoaderData,
} from "@remix-run/react";
import "./tailwind.css";
import { getToast } from "remix-toast";
import { LinksFunction, LoaderFunctionArgs } from "@remix-run/node";
import { useEffect } from "react";
import { Toaster, toast as notify } from "sonner";

export const loader = async ({ request }: LoaderFunctionArgs) => {
  const { toast, headers } = await getToast(request);
  return json({ toast }, { headers });
}

export function Layout({ children }: { children: React.ReactNode }) {
  const data = useLoaderData<typeof loader>();

  useEffect(() => {
    if (data?.toast?.type === "error") {
      notify.error(data.toast.message);
    }
    if (data?.toast?.type === "success") {
      notify.success(data.toast.message);
    }
  }, [data?.toast]);
  return (
    <html lang="en">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <Meta />
        <Links />
      </head>
      <body>
        {children}
        <ScrollRestoration />
        <Scripts />
        <Toaster />
      </body>
    </html>
  );
}

export default function App() {
  return <Outlet />;
}
