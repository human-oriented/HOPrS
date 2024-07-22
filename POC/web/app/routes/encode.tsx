import { MetaFunction, json, ActionFunction, LinksFunction } from "@remix-run/node";
import { Form, Link, useFetcher } from "@remix-run/react";
import { useEffect, useState } from "react";
import * as Icon from 'react-feather';
import { jsonWithSuccess, jsonWithError } from "remix-toast";
import { LoadingButton } from "~/components/LoadingButton";

export const meta: MetaFunction = () => {
  return [
    { title: "HOPrS" },
    { name: "description", content: "HOPrS - Human oriented proof standard" },
  ];
};

export const action: ActionFunction = async ({ request }) => {
    const formData = await request.formData();
    console.log('formadata', formData)
    const file = formData.get('file');
    const depth = formData.get('depth');
    const algorithm = formData.get('algorithm');
  
    if (!file || typeof file === 'string') {
      return json({ error: 'File is required' }, { status: 400 });
    }
  
    const externalApiUrl = `${process.env.API_URL}/hoprs/encode`; // Replace with your external API URL
    const externalFormData = new FormData();
    externalFormData.append('file', file, file.name);
    externalFormData.append('depth', depth);
    externalFormData.append('algorithm', algorithm);
    console.log('external form data', externalFormData)
    try {
      const response = await fetch(externalApiUrl, {
        method: 'POST',
        body: externalFormData,
        
      });
  
      if (!response.ok) {
        throw new Error('Failed to upload file to external API');
      }
  
      const responseData = await response.text();
      console.log('response', responseData)
      return jsonWithSuccess({ success: true, blob: responseData }, "Image encoded successfully!");
    } catch (error) {
        console.error(error)
      return jsonWithError({ error: (error as Error).message }, error.message);
    }
  };

export default function Encode() {
    const fetcher = useFetcher()
    const [depth, setDepth] = useState(5)
    const [image, setImage] = useState('')
    const [download, setDownload] = useState('')
    const [loading, setLoading] = useState(false)

useEffect(() => {
    if (fetcher?.data?.blob) {
        var blobObj = new Blob([fetcher.data.blob], { type: "text/csv" });
        setDownload(window.URL.createObjectURL(blobObj))
    }

    if(fetcher.data && fetcher.data.error) {
        console.error('fetcher error', fetcher.data.error)
    }
}, [fetcher?.data])

useEffect(() => {
    if (fetcher?.state != "idle") {
        setLoading(true)
    } else {
        setLoading(false)
    }
}, [fetcher?.state])

  return (
    <>
    
    <div className="p-5 container m-auto text-center items-center justify-center flex flex-col flex-stretch min-h-screen pt-20">
      <div className="intro text-gray-500 lg:w-2/3 mx-auto mb-5 text-sm">
        <h1 className="font-bold text-3xl text-black">Image Encoder</h1>
        <p className="my-3">This tool will encode any image and generate quad tree file which can be used for image comparison.</p>
        <p>To compare and image with the generated quad tree file you can use our <Link className="underline" to="/compare">image comparison tool</Link>.</p>
      </div>
      <fetcher.Form method="POST" encType="multipart/form-data" className="flex flex-col w-full">
        <div className="files flex flex-row items-center">
            <div className="w-1/2 p-5">
                <div className="flex items-center justify-center w-full">
                    <label htmlFor="file" className="flex flex-col items-center justify-center w-full h-64 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-gray-50 dark:hover:bg-gray-800 dark:bg-gray-700 hover:bg-gray-100 dark:border-gray-600 dark:hover:border-gray-500 dark:hover:bg-gray-600 p-3">
                        <div className="flex flex-col items-center justify-center pt-5 pb-6 text-black w-full">
                            <Icon.Image size={40} />
                            {image ?
                            <>
                                <p className="text-sm text-gray-500 w-2/3 my-3">{image?.name}</p>
                                <p className="bg-black px-5 py-2 text-white rounded-full text-sm">Change</p>
                            </>
                            :
                            <>
                                <p className="my-2 font-bold text-black">Image upload</p>
                                <p className="mb-2 text-sm text-gray-500 dark:text-gray-400"><span className="font-semibold">Click to upload</span> or drag and drop</p>
                                <p className="text-xs text-gray-500 dark:text-gray-400">HEIC, PNG, JPG or GIF (MAX. 4032x3024px)</p>
                            </>
                            }
                        </div>
                        <input id="file" name="file" type="file" onChange={e => setImage(e?.target?.files[0])} className="sr-only" />
                    </label>
                </div> 
            </div>
            <div className="w-1/2 p-5">
                <div className="relative mb-6 text-left">
                    <label htmlFor="depth" className="text-xs text-left w-full font-bold flex flex-row">Quad tree depth <Icon.Info className="ml-1" size={15} /></label>
                    <input id="depth" name="depth" type="range" defaultValue={depth} onChange={e => setDepth(parseInt(e.target.value))} min="1" max="10" className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-black dark:bg-gray-700" />
                    <ul className="flex justify-between w-full px-[10px] text-xs text-gray-500">
                        <li className="flex justify-center relative"><span className="absolute">1</span></li>
                        <li className="flex justify-center relative"><span className="absolute">2</span></li>
                        <li className="flex justify-center relative"><span className="absolute">3</span></li>
                        <li className="flex justify-center relative"><span className="absolute">4</span></li>
                        <li className="flex justify-center relative"><span className="absolute">5</span></li>
                        <li className="flex justify-center relative"><span className="absolute">6</span></li>
                        <li className="flex justify-center relative"><span className="absolute">7</span></li>
                        <li className="flex justify-center relative"><span className="absolute">8</span></li>
                        <li className="flex justify-center relative"><span className="absolute">9</span></li>
                        <li className="flex justify-center relative"><span className="absolute">10</span></li>
                    </ul>
                </div>
                <div className="text-left flex flex-col py-5">
                    <label htmlFor="labels-range-input" className="text-xs text-left w-full font-bold flex flex-row">Hash Algorithm <Icon.Info className="ml-1" size={15} /></label>
                    <select name="algorithm" className="bg-gray-100 p-2 rounded-lg my-2">
                        <option value="pdq">PDQ</option>
                    </select>
                </div>
                
            </div>  
        </div>
        
        <div className="options flex flex-col text-left items-center px-5">
            <div className="heading flex flex-row items-center">
                <button className="font-bold text-sm" type="button">Advanced Options</button>
                <Icon.ChevronDown size={15} />
            </div>
            
            <div className="options-inner hidden">
                
            </div>
        </div>
        <div className="buttons my-10">
            <LoadingButton loading={loading} width={300} className="bg-black inline-block mb-5">Encode Image</LoadingButton>
            {fetcher.data && fetcher.data.success ? 
                <a className="bg-green-500 mt-5 inline-block hover:opacity-90 text-white font-bold px-10 py-[18px] rounded-tl-2xl rounded-br-2xl  min-w-[300px] lg:ml-2" href={download}>Download Quad tree file</a>
            :
                <a className="bg-gray-300 mt-5 inline-block cursor-not-allowed text-white font-bold px-10 py-[18px] rounded-tl-2xl rounded-br-2xl min-w-[300px] lg:ml-2" href={null}>Download Quad tree file</a>
            }
        </div>
        
      </fetcher.Form>
    </div>
    </>
  );
}
