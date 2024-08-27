import type { MetaFunction } from "@remix-run/node";
import { Link } from "@remix-run/react";

export const meta: MetaFunction = () => {
  return [
    { title: "New Remix App" },
    { name: "description", content: "Welcome to Remix!" },
  ];
};

export default function Index() {
  return (
    <div className="text-2xl font-bold p-5 container m-auto text-center items-center justify-center flex flex-col h-screen">
      <div className="mb-10">
        <h1 className="font-bold font-serif text-[80px] mb-10">HOPrS.</h1>
        <p className="text-sm font-normal text-left -mt-4">[Hop&bull;pers] <span className="italic">noun.</span></p>
        <p className="text-sm text-left">Human oriented proof standard.</p>
      </div>
      <Link className="my-3" to="/encode">Encode</Link>
      <Link className="mb-3" to="/demo-compare">Compare</Link>
      
    </div>
  );
}
